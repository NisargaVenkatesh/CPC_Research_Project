import torch.nn as nn

class AutoregressiveModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        output, _ = self.gru(x)
        return output