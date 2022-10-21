import numpy as np
import torch
import torch.nn as nn
import models
from .models import register

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

@register('generator')
class Generator(nn.Module):
    """
    To generate latent distribution parameters
    """

    def log_var_block(self, input_dim, output_dim=512):
        block = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.Dropout(0.25),
            torch.nn.Sigmoid()

        )
        return block

    def mu_block(self, input_dim, output_dim=512):
        # import pdb; pdb.set_trace()
        block = torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.Sigmoid()
        )
        return block

    def __init__(self, g_input):
        super(Generator, self).__init__()
        # self.input_dim = 512 if g_input == 'resnet12' else 1600
        self.input_dim = g_input
        # self.fc_mu = torch.nn.Linear(self.input_dim, 512)
        self.fc_mu = self.mu_block(self.input_dim)
        self.fc_var = self.log_var_block(self.input_dim)
        self.dropout = torch.nn.Dropout(0.25)
        self.apply(weight_init)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        # mu = self.dropout(mu)
        log_var = self.fc_var(x)
        # log_var = self.dropout(log_var)

        return mu, log_var
