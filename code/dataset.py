import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

USER_ID = 'user_id'
ITEM_ID = 'item_id'
RATING = 'rating'


class TrainDataset(Dataset):
    def __init__(self, train_data_path):
        train_data_df = pd.read_csv(train_data_path, sep='\t')

        self.data_df = train_data_df[[USER_ID, ITEM_ID]]

        self.pos_inter_num = len(train_data_df)
        self.pos_user_ids = train_data_df.user_id
        self.pos_item_ids = train_data_df.item_id
        self.user_num = train_data_df.user_id.nunique()
        self.item_num = train_data_df.item_id.nunique()
        self.user2posItem_dict = self.get_user2posItem_dict(train_data_df)

    def get_user2posItem_dict(self, df):
        user2posItem_dict = df.groupby(USER_ID)[ITEM_ID].apply(list).to_dict()
        return user2posItem_dict

    def __len__(self):
        return self.pos_inter_num
    
    def __getitem__(self, index):
        pos_user_id = self.pos_user_ids[index]
        pos_item_id = self.pos_item_ids[index]
        return pos_user_id, pos_item_id

    def train_collect_fn(self, batch_data):
        user_ids = []
        batch_size = len(batch_data)

        user_ids = np.array([data[0] for data in batch_data], dtype=np.int64)
        pos_item_ids = np.array([data[1] for data in batch_data], dtype=np.int64)

        sample_neg_index = np.zeros(batch_size, dtype=np.int64)
        check_index = np.arange(batch_size)
        while len(check_index) > 0:
            sample_neg_index[check_index] = np.random.randint(0, self.item_num, size=len(check_index))
            check_index = np.array([i for i, u, sample_item in zip(check_index, user_ids[check_index], sample_neg_index[check_index]) if sample_item in self.user2posItem_dict[u]])

        return torch.LongTensor(user_ids), torch.LongTensor(pos_item_ids), torch.LongTensor(sample_neg_index)


class UserTrainDataset(Dataset):
    def __init__(self, train_data_path):
        train_data_df = pd.read_csv(train_data_path, sep='\t')

        self.data_df = train_data_df[[USER_ID, ITEM_ID]]

        self.user_num = train_data_df.user_id.nunique()
        self.item_num = train_data_df.item_id.nunique()
        self.user2posItem_dict = self.get_user2posItem_dict(train_data_df)

        self.user_list = torch.arange(self.user_num)

    def get_user2posItem_dict(self, df):
        user2posItem_dict = df.groupby(USER_ID)[ITEM_ID].apply(list).to_dict()
        return user2posItem_dict

    def __len__(self):
        return self.user_num
    
    def __getitem__(self, index):
        return self.user_list[index]


class TestDataset(Dataset):
    def __init__(self, test_data_path, *history_data_df):
        test_data_df = pd.read_csv(test_data_path, sep='\t')

        self.data_df = test_data_df[[USER_ID, ITEM_ID]]
        self.user_ids = test_data_df.user_id.unique()
        self.user2posItem_dict = self.get_user2posItem_dict(test_data_df)
        self.user2hisItem_dict = self.get_user2hisItem_dict(history_data_df)

    def get_user2posItem_dict(self, df):
        user2posItem_dict = df.groupby(USER_ID)[ITEM_ID].apply(list).to_dict()
        return user2posItem_dict

    def get_user2hisItem_dict(self, history_data_df):
        if len(history_data_df) == 0:
            return defaultdict(list)
        df = pd.concat(history_data_df)
        user2hisItem_dict = df.groupby(USER_ID)[ITEM_ID].apply(list).to_dict()
        return user2hisItem_dict            
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        return user_id, self.user2hisItem_dict[user_id], self.user2posItem_dict[user_id]

    def test_collect_fn(self, batch_data):
        user_ids = []
        hist_items = []
        pos_items = []
        for data in batch_data:
            user_ids.append(data[0])
            hist_items.append(data[1])
            pos_items.append(data[2])

        return torch.LongTensor(np.array(user_ids, dtype=np.int64)), hist_items, pos_items


def data_preparation(config):
    if config['model'] in ['CDAE', 'ACAE']:
        train_dataset = UserTrainDataset(config['train_data_path'])
        train_dataloader = DataLoader(train_dataset, batch_size=config['train_bs'], pin_memory=False, shuffle=True, num_workers=config['num_workers'])
    else:
        train_dataset = TrainDataset(config['train_data_path'])
        train_dataloader = DataLoader(train_dataset, batch_size=config['train_bs'], pin_memory=False, shuffle=True, num_workers=config['num_workers'], collate_fn=train_dataset.train_collect_fn)
    
    valid_dataset = TestDataset(config['valid_data_path'], train_dataset.data_df)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['test_bs'], pin_memory=False, shuffle=False, num_workers=config['num_workers'], collate_fn=valid_dataset.test_collect_fn)
    
    test_dataset = TestDataset(config['test_data_path'], train_dataset.data_df, valid_dataset.data_df)
    test_dataloader = DataLoader(test_dataset, batch_size=config['test_bs'], pin_memory=False, shuffle=False, num_workers=config['num_workers'], collate_fn=test_dataset.test_collect_fn)
    
    return train_dataloader, valid_dataloader, test_dataloader