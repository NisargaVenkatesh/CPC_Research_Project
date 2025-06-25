import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, T]
        return self.conv(x).permute(0, 2, 1)  # [B, T', D]
