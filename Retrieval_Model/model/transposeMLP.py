import torch.nn as nn
import torch.nn.functional as F
import torch

class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()

        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.act = Swish(inplace=True)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            #             nn.LayerNorm(hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransposeMLP(nn.Module):
    def __init__(self, dim, seq_len, token_dim, channel_dim):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, token_dim),
        )

        self.channel_mix = nn.Sequential(
            #             nn.LayerNorm(seq_len),
            FeedForward(seq_len, channel_dim),
        )

    def forward(self, x):
        shortcut = x
        x = self.token_mix(x)
        #         x = x + shortcut
        x = x.permute(0, 2, 1)
        x = self.channel_mix(x)
        x = x.permute(0, 2, 1)
        return x + shortcut


class Net(nn.Module):
    def __init__(
            self,
            seq_len,
            d_model,
            token_dim,
            channel_dim,
            represent_dim,
            n_blocks,
            n_classes,
    ):
        super().__init__()
        self.seq_len = seq_len + 1
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.transposemlp = nn.ModuleList([])
        for _ in range(n_blocks):
            self.transposemlp.append(
                TransposeMLP(dim=self.d_model, seq_len=self.seq_len, token_dim=token_dim, channel_dim=channel_dim))
        self.represent = nn.Linear(d_model, represent_dim)
        self.bn = nn.BatchNorm1d(represent_dim, momentum=0.9)
        self.head = nn.Linear(represent_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = x.view(n_samples, -1, self.d_model)
        cls_token = self.cls_token.expand(n_samples, 1, -1)
        x = torch.cat((cls_token, x), dim=1)
        for transposemlp in self.transposemlp:
            x = transposemlp(x)
        x = self.layer_norm(x)
        cls_token_final = x[:, 0]
        fea = self.bn(self.represent(cls_token_final))
        x = self.head(fea)
        if self.training:
            return fea, x
        else:
            return fea
