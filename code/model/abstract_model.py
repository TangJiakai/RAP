import logging

from utils import set_color
import torch
import numpy as np


class AbstractRecommender(torch.nn.Module):
    def __init__(self):
        self.logger = logging.getLogger()
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, batch_data):
        raise NotImplementedError

    def predict(self, batch_data):
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + set_color("\nTrainable parameters", "blue")
            + f": {params}"
        )