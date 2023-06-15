'''
RAP
'''


from model.abstract_model import AbstractRecommender
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from model_init import module_inititialization
from loss import BPRLoss

CHUNK_SIZE_FOR_SPMM = 1000000


class RAP(AbstractRecommender):
    def __init__(self, config, dataset):
        super(RAP, self).__init__()

        self.device = config['device']
        self.embedding_size = config['embedding_size']
        self.eps = config['eps']
        self.RAP_layer_num = config['RAP_layer_num']
        self.beta = config['beta']
        self.adv_loss_weight = config['adv_loss_weight']

        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        self.loss_fun = BPRLoss

        self.interaction_matrix = self.create_coo_interaction_matrix(dataset)
        self.adj_matrix = self.get_adj_mat(self.interaction_matrix)
        self.norm_adj_matrix = self.get_norm_mat(self.adj_matrix).to(self.device)

        self.for_learning_adj()
        
        self.restore_user_e = None
        self.restore_item_e = None

        self.apply(lambda module: module_inititialization(module))

    def for_learning_adj(self):
        self.adj_indices = self.norm_adj_matrix.indices()
        self.adj_shape = self.norm_adj_matrix.shape
        self.adj = self.norm_adj_matrix

        inter_data = torch.FloatTensor(self.interaction_matrix.data).to(self.device)
        inter_user = torch.LongTensor(self.interaction_matrix.row).to(self.device)
        inter_item = torch.LongTensor(self.interaction_matrix.col).to(self.device)
        inter_mask = torch.stack([inter_user, inter_item], dim=0)

        self.inter_spTensor = torch.sparse.FloatTensor(inter_mask, inter_data, self.interaction_matrix.shape).coalesce()
        self.inter_spTensor_t = self.inter_spTensor.t().coalesce()

        self.inter_indices = self.inter_spTensor.indices()
        self.inter_shape = self.inter_spTensor.shape

    def create_coo_interaction_matrix(self, dataset):
        src = dataset.pos_user_ids
        tgt = dataset.pos_item_ids
        data = np.ones(len(dataset))

        mat = coo_matrix(
            (data, (src, tgt)), shape=(self.user_num, self.item_num)
        ).astype(np.float32)

        return mat

    def get_adj_mat(self, inter_M, data=None):
        if data is None:
            data = [1] * inter_M.data
        inter_M_t = inter_M.transpose()
        A = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.user_num), data))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.user_num, inter_M_t.col), data)))
        A._update(data_dict)  # dok_matrix
        return A

    def get_norm_mat(self, A):
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape)).coalesce()

        return SparseL

    def get_sim_adj(self):
        sim_mat = self.get_sim_mat()
        sim_adj = self.inter2adj(sim_mat)

        # pruning
        sim_value = torch.div(torch.add(sim_adj.values(), 1), 2)
        pruned_sim_value = torch.where(sim_value < self.beta, torch.zeros_like(sim_value),
                                       sim_value) if self.beta > 0 else sim_value
        pruned_sim_adj = torch.sparse.FloatTensor(sim_adj.indices(), pruned_sim_value, self.adj_shape).coalesce()
        self.pruned_sim_adj = pruned_sim_adj

        # normalize
        pruned_sim_indices = pruned_sim_adj.indices()
        diags = torch.sparse.sum(pruned_sim_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -1)
        diag_lookup = diags[pruned_sim_indices[0, :]]

        pruned_sim_adj_value = pruned_sim_adj.values()
        normal_sim_value = torch.mul(pruned_sim_adj_value, diag_lookup)
        normal_sim_adj = torch.sparse.FloatTensor(pruned_sim_indices, normal_sim_value,
                                                  self.adj_shape).to(self.device).coalesce()

        return normal_sim_adj

    def sp_cos_sim(self, a, b):
        eps = 1e-8
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

        L = self.inter_indices.shape[1]
        sims = torch.zeros(L, dtype=a.dtype).to(self.device)
        for idx in range(0, L, CHUNK_SIZE_FOR_SPMM):
            batch_indices = self.inter_indices[:, idx:idx + CHUNK_SIZE_FOR_SPMM]

            a_batch = torch.index_select(a_norm, 0, batch_indices[0, :])
            b_batch = torch.index_select(b_norm, 0, batch_indices[1, :])

            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE_FOR_SPMM] = dot_prods

        return torch.sparse_coo_tensor(self.inter_indices, sims, size=self.interaction_matrix.shape,
                                       dtype=sims.dtype).coalesce()

    def get_sim_mat(self):
        user_feature = torch.sparse.mm(self.inter_spTensor, self.item_embedding.weight)
        item_feature = torch.sparse.mm(self.inter_spTensor_t, self.user_embedding.weight)
        sim_inter = self.sp_cos_sim(user_feature, item_feature)
        return sim_inter

    def inter2adj(self, inter):
        inter_t = inter.t().coalesce()
        data = inter.values()
        data_t = inter_t.values()
        adj_data = torch.cat([data, data_t], dim=0)
        adj = torch.sparse.FloatTensor(self.adj_indices, adj_data, self.adj_shape).to(self.device).coalesce()
        return adj

    def forward(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]

        self.adj = self.norm_adj_matrix if self.beta < 0.0 else self.get_sim_adj()
        for _ in range(self.RAP_layer_num):
            all_embeddings = torch.sparse.mm(self.adj, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.user_num, self.item_num])
        return user_all_embeddings, item_all_embeddings

    def calculate_bpr_loss(self, u_embeddings, pos_embeddings, neg_embeddings, mean=True):
        pos_pred_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_pred_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        loss = self.loss_fun(pos_pred_scores, neg_pred_scores, mean)
        return loss

    def calculate_corruption(self, loss):
        corruption_scale = 1 / (1 + loss.detach())
        corruption_scale = torch.clip(corruption_scale, min=0, max=1)
        return corruption_scale[:, None]

    def calculate_loss(self, batch_data): 
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        users, pos_items, neg_items = batch_data
        
        # calculate delta embedding
        user_all_embeddings, item_all_embeddings = self.forward()
        user_embs, pos_item_embs, neg_item_embs = user_all_embeddings[users], item_all_embeddings[pos_items], item_all_embeddings[neg_items]
        normal_loss = self.calculate_bpr_loss(user_embs, pos_item_embs, neg_item_embs, mean=False)

        user_embs_grad, pos_item_embs_grad, neg_item_embs_grad = autograd.grad(normal_loss.mean(), [user_embs, pos_item_embs, neg_item_embs], retain_graph=True)
        user_embs_grad.requires_grad, pos_item_embs_grad.requires_grad, neg_item_embs_grad.requires_grad = False, False, False

        corruption_scale = self.calculate_corruption(normal_loss)

        delta_user_embs = F.normalize(user_embs_grad) * self.eps * corruption_scale
        delta_pos_item_embs = F.normalize(pos_item_embs_grad, dim=-1) * self.eps * corruption_scale
        delta_neg_item_embs = F.normalize(neg_item_embs_grad, dim=-1) * self.eps * corruption_scale

        # Adv Loss
        adv_user_embs = user_embs + delta_user_embs
        adv_pos_item_embs = pos_item_embs + delta_pos_item_embs
        adv_neg_item_embs = neg_item_embs + delta_neg_item_embs

        adv_loss = self.adv_loss_weight * self.calculate_bpr_loss(adv_user_embs, adv_pos_item_embs, adv_neg_item_embs)

        return normal_loss.mean(), adv_loss

    def full_rank(self, user_ids):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        user_embedding = self.restore_user_e[user_ids]
        all_item_embedding = self.restore_item_e
        pred_scores = torch.matmul(user_embedding, all_item_embedding.T)
        return pred_scores