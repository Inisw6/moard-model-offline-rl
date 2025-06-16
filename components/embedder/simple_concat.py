from typing import Any, Dict, List

from components.core.base import BaseEmbedder
from components.registry import register


@register("simple_concat")
class SimpleConcatEmbedder(BaseEmbedder):
    """SimpleConcatEmbedder 클래스.

    지정된 user_embedder와 content_embedder를 레지스트리에서 생성하여
    유저/콘텐츠 임베딩을 위임합니다.

    Attributes:
        user_embedder: 사용자 임베딩 객체 (BaseUserEmbedder).
        content_embedder: 콘텐츠 임베딩 객체 (BaseContentEmbedder).
        content_types (List[str]): 처리 가능한 콘텐츠 타입 목록.
        num_content_types (int): 콘텐츠 타입 수.
        type_to_idx_map (Dict[str, int]): 콘텐츠 타입 → 인덱스 매핑.
        user_dim (int): 사용자 임베딩 차원.
        content_dim (int): 콘텐츠 임베딩 차원.
    """

    def __init__(
        self,
        user_embedder: Dict[str, Any],
        content_embedder: Dict[str, Any],
    ) -> None:
        """SimpleConcatEmbedder 생성자.

        Args:
            user_embedder (Dict[str, Any]):
                {'type': str, 'params': dict} 형태의 사용자 임베더 설정.
            content_embedder (Dict[str, Any]):
                {'type': str, 'params': dict} 형태의 콘텐츠 임베더 설정.
        """
        from components.registry import make

        # 레지스트리에서 실제 임베더 인스턴스 생성
        self.user_embedder = make(user_embedder["type"], **user_embedder["params"])
        self.content_embedder = make(
            content_embedder["type"], **content_embedder["params"]
        )

        # 콘텐츠 타입 정보 및 매핑
        self.content_types: List[str] = self.content_embedder.content_types
        self.num_content_types: int = len(self.content_types)
        self.type_to_idx_map: Dict[str, int] = {
            t: i for i, t in enumerate(self.content_types)
        }

        # 임베딩 차원 정보
        self.user_dim: int = self.user_embedder.user_dim
        self.content_dim: int = self.content_embedder.content_dim

        super().__init__(self.user_embedder, self.content_embedder)
