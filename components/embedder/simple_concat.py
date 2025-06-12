from components.core.base import BaseEmbedder
from components.registry import register


@register("simple_concat")
class SimpleConcatEmbedder(BaseEmbedder):
    """SimpleUserEmbedder와 SimpleContentEmbedder를 결합한 임베더.

    - 지정된 user_embedder와 content_embedder에 위임하여 임베딩 수행.
    """

    def __init__(self, user_embedder: dict, content_embedder: dict) -> None:
        """SimpleConcatEmbedder 생성자.

        Args:
            user_embedder (dict): {"type": str, "params": dict}
            content_embedder (dict): {"type": str, "params": dict}
        """
        from components.registry import make

        # 레지스트리에서 user 임베더 인스턴스 생성
        self.user_embedder = make(user_embedder["type"], **user_embedder["params"])
        # 레지스트리에서 content 임베더 인스턴스 생성
        self.content_embedder = make(
            content_embedder["type"], **content_embedder["params"]
        )

        # content_embedder로부터 콘텐츠 타입 목록과 매핑 정보 가져오기
        self.content_types = self.content_embedder.content_types
        self.num_content_types = len(self.content_types)
        self.type_to_idx_map = {t: i for i, t in enumerate(self.content_types)}

        # user_dim과 content_dim 정보 저장
        self.user_dim = self.user_embedder.user_dim
        self.content_dim = self.content_embedder.content_dim

        # BaseEmbedder 초기화 (user_embedder, content_embedder를 인자로 전달)
        super().__init__(self.user_embedder, self.content_embedder)
