import torch.nn as nn
import torch
import torch.nn.functional as F


class Fuse_add_encoder_rep(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_review = nn.Linear(hidden_size, hidden_size)
        self.linear_seq_rep = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, seq_rep, exp_rep, mask):
        # exp_rep = self.linear_review(exp_rep)
        # seq_rep = self.linear_seq_rep(seq_rep)
        alpha = torch.matmul(exp_rep, seq_rep.unsqueeze(1).transpose(1, 2)).squeeze(-1)
        alpha = alpha.masked_fill(mask == 0, -1e12)
        alpha = F.softmax(alpha, dim=-1)
        fuse_rep = alpha.unsqueeze(-1) * seq_rep.unsqueeze(1) + exp_rep
        return fuse_rep
