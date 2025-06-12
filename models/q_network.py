import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Q-Value를 예측하는 MLP 네트워크.

    Args:
        user_dim (int): 사용자 임베딩 벡터 차원.
        content_dim (int): 콘텐츠 임베딩 벡터 차원.
        hidden_dim (int, optional): 은닉층 크기. 기본값은 128.

    Input:
        user (torch.Tensor): [batch_size, user_dim]
        content (torch.Tensor): [batch_size, content_dim]

    Output:
        torch.Tensor: [batch_size, 1] Q-value
    """

    def __init__(self, user_dim: int, content_dim: int, hidden_dim: int = 128) -> None:
        """QNetwork 클래스 생성자.

        Args:
            user_dim (int): 사용자 임베딩 벡터 차원.
            content_dim (int): 콘텐츠 임베딩 벡터 차원.
            hidden_dim (int, optional): 은닉층 크기. 기본값은 128.
        """
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
        """사용자/콘텐츠 벡터를 받아 Q-value를 예측합니다.

        Args:
            user (torch.Tensor): [batch_size, user_dim]
            content (torch.Tensor): [batch_size, content_dim]

        Returns:
            torch.Tensor: [batch_size, 1] Q-value

        Raises:
            ValueError: 배치 사이즈 또는 차원이 일치하지 않을 때.
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
<<<<<<< HEAD
<<<<<<< HEAD
=======



class DuelingQNetwork(nn.Module):
    """Dueling 구조의 Q-Value 예측 네트워크.

    Args:
        user_dim (int): 사용자 임베딩 벡터 차원.
        content_dim (int): 콘텐츠 임베딩 벡터 차원.
        hidden_dim (int, optional): 은닉층 크기. 기본값은 128.

    Input:
        user (torch.Tensor): [batch_size, user_dim]
        content (torch.Tensor): [batch_size, content_dim]

    Output:
        torch.Tensor: [batch_size, 1] Q-value
    """

    def __init__(self, user_dim: int, content_dim: int, hidden_dim: int = 128) -> None:
        """
        DuelingQNetwork 클래스 생성자.

        Args:
            user_dim (int): 사용자 임베딩 벡터 차원.
            content_dim (int): 콘텐츠 임베딩 벡터 차원.
            hidden_dim (int, optional): 은닉층 크기. 기본값은 128.
        """
        super().__init__()
        self.user_dim = user_dim
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim

        # 사용자와 콘텐츠 임베딩을 합쳐서 처리하는 공유 네트워크
        self.shared = nn.Sequential(
            nn.Linear(user_dim + content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Value stream: 상태의 가치를 추정
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Advantage stream: 각 행동의 상대적 우수성을 추정
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """
        사용자/콘텐츠 벡터를 받아 Dueling 구조로 Q-value를 예측합니다.

        Args:
            user (torch.Tensor): [batch_size, user_dim]
            content (torch.Tensor): [batch_size, content_dim]

        Returns:
            torch.Tensor: [batch_size, 1] Q-value
        """
        x = torch.cat([user, content], dim=-1)  # 입력 벡터 결합
        h = self.shared(x)  # 공유 네트워크 통과
        v = self.value_stream(h)  # Value stream 결과
        a = self.adv_stream(h)  # Advantage stream 결과
        # Dueling 구조의 Q-value 계산
        return v + (a - a.mean(dim=0, keepdim=True))
<<<<<<< HEAD
>>>>>>> 2688699 (feat: Dueling DQN 에이전트 추가)
=======
>>>>>>> 51d0e23 (feat: agent 및 q network 분리)
=======
>>>>>>> 2f03202 (feat: Dueling DQN 에이전트 추가)
