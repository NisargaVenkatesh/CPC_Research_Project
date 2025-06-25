import torch
import torch.nn as nn
import torch.nn.functional as F

class CPCLoss(nn.Module):
    def __init__(self, prediction_steps):
        super().__init__()
        self.prediction_steps = prediction_steps
        self.Wk = nn.ModuleList([nn.Linear(512, 512) for _ in range(prediction_steps)])

    def forward(self, z, c):
        B, T, D = z.size()
        loss = 0.0
        for k in range(1, self.prediction_steps + 1):
            if T - k <= 0:
                continue
            z_pred = self.Wk[k - 1](c[:, :-k])  # [B, T-k, D]
            z_true = z[:, k:]                  # [B, T-k, D]
            logits = torch.bmm(z_pred, z_true.transpose(1, 2))  # [B, T-k, T-k]
            labels = torch.arange(T - k).unsqueeze(0).expand(B, -1).to(z.device)
            loss += F.cross_entropy(logits, labels)
        return loss / self.prediction_steps