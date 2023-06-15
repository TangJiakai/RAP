import torch.nn as nn


def module_inititialization(module, method='normal'):
    if method == 'normal':
        xavier_normal_initialization(module)
    elif method == 'uniform':
        xavier_uniform_initialization(module)


def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        if module.weight.requires_grad:
            nn.init.xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        if module.weight.requires_grad:
            nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None and module.bias.requires_grad:
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

def xavier_uniform_initialization(module):
    if isinstance(module, nn.Embedding):
        if module.weight.requires_grad:
            nn.init.xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        if module.weight.requires_grad:
            nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None and module.bias.requires_grad:
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)