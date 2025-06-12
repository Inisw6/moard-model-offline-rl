import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Q-Value를 예측하는 기본 MLP 네트워크 (DQN)."""
    def __init__(self, user_dim: int, content_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.user_dim = user_dim
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(user_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        if user.shape[0] != content.shape[0]:
            raise ValueError(
                f"Batch size mismatch: user.shape={user.shape}, content.shape={content.shape}"
            )
        if user.shape[1] != self.user_dim or content.shape[1] != self.content_dim:
            raise ValueError(
                f"Input dim mismatch: user_dim={user.shape[1]}, expected={self.user_dim}; content_dim={content.shape[1]}, expected={self.content_dim}"
            )
        x = torch.cat([user, content], dim=1)
        return self.net(x)  # [batch, 1]

class DuelingQNetwork(nn.Module):
    """Dueling 구조의 Q-Value 예측 네트워크."""
    def __init__(self, user_dim: int, content_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.user_dim = user_dim
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(user_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, user: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        x = torch.cat([user, content], dim=-1)
        h = self.shared(x)
        v = self.value_stream(h)
        a = self.adv_stream(h)
        return v + (a - a.mean(dim=0, keepdim=True))