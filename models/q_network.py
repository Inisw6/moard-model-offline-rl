import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, user_dim: int, content_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(user_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        # user:   [batch, user_dim]
        # content:[batch, content_dim]
        x = torch.cat([user, content], dim=1)
        return self.net(x)  # [batch, 1]
