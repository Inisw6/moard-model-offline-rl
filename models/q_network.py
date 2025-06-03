import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q-Value를 예측하는 MLP 네트워크.

    Args:
        user_dim (int): 사용자 임베딩 벡터 차원.
        content_dim (int): 콘텐츠 임베딩 벡터 차원.
        hidden_dim (int, optional): 은닉층 크기. 기본값=128.

    입력:
        user:   [batch_size, user_dim] (Tensor)
        content:[batch_size, content_dim] (Tensor)

    출력:
        Q-value: [batch_size, 1] (Tensor)
    """

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
        """
        사용자/콘텐츠 벡터를 받아 Q-value를 예측합니다.

        Args:
            user (torch.Tensor): [batch_size, user_dim]
            content (torch.Tensor): [batch_size, content_dim]

        Returns:
            torch.Tensor: [batch_size, 1] Q-value
        """
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
