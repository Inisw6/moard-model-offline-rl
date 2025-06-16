from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def enforce_type_constraint(
    q_values: Dict[str, List[float]], top_k: int = 6
) -> List[Tuple[str, int]]:
    """타입별 Q값에서 상위 top_k개를 뽑되, 모든 타입이 최소 1개씩 포함되도록 보정합니다.

    1) 모든 (type, index, q_value) 튜플을 평탄화하여 정렬
    2) 상위 top_k개를 선택
    3) 선택에 누락된 타입이 있으면, 해당 타입의 최상위 아이템을
       최하위 아이템과 교체한 뒤 재정렬

    Args:
        q_values (Dict[str, List[float]]):
            타입별 Q값 리스트.
            예: {'youtube': [q0, q1, ...], 'blog': [...], 'news': [...]}
        top_k (int): 최종 선택할 아이템 개수.

    Returns:
        List[Tuple[str, int]]:
            (콘텐츠 타입, 해당 타입 내 인덱스) 쌍의 리스트, 길이는 top_k.
    """
    # 모든 항목 평탄화
    all_items: List[Tuple[str, int, float]] = [
        (ctype, idx, qv)
        for ctype, q_list in q_values.items()
        for idx, qv in enumerate(q_list)
    ]

    # Q값 내림차순 정렬
    all_items.sort(key=lambda x: x[2], reverse=True)

    # 상위 top_k개 선택
    selected = all_items[:top_k]
    included_types = {ctype for ctype, _, _ in selected}
    missing_types = set(q_values) - included_types

    # 누락된 타입 보정
    for mtype in missing_types:
        q_list = q_values.get(mtype, [])
        if not q_list:
            continue
        best_idx = int(np.argmax(q_list))
        best_q = q_list[best_idx]

        # 이미 선택되지 않았다면, 최하위와 교체
        if not any(t == mtype and i == best_idx for t, i, _ in selected):
            selected[-1] = (mtype, best_idx, best_q)
            selected.sort(key=lambda x: x[2], reverse=True)

    # (type, index) 형태로 반환
    return [(t, i) for t, i, _ in selected]


def compute_all_q_values(
    state: Any,
    cand_dict: Dict[str, List[Dict[str, Any]]],
    embedder: Any,
    agent: Any,
    emb_cache: Optional[Dict[Any, Any]] = None,
) -> Dict[str, List[float]]:
    """상태와 후보 콘텐츠 딕셔너리로부터 타입별 Q값 리스트를 일괄 계산합니다.

    1) 모든 콘텐츠 임베딩을 계산/캐싱
    2) 하나의 배치로 Q-network에 넣어 Q값 추론
    3) 결과를 원래 타입별 리스트로 분배

    Args:
        state (Any): 현재 상태 벡터 (예: np.ndarray).
        cand_dict (Dict[str, List[Dict[str, Any]]]):
            타입별 후보 콘텐츠 목록.
        embedder (Any): embed_content 메서드를 가진 임베딩 객체.
        agent (Any): q_net 및 device 속성을 가진 에이전트 객체.
        emb_cache (Optional[Dict[Any, Any]]):
            콘텐츠 ID → 임베딩 캐시, 없으면 None.

    Returns:
        Dict[str, List[float]]:
            타입별 Q값 리스트 사전.
    """
    # 결과 사전 초기화
    q_values: Dict[str, List[float]] = {ctype: [] for ctype in cand_dict}
    all_embs, type_indices = [], []

    # 임베딩 수집 및 캐시
    for type_idx, (ctype, items) in enumerate(cand_dict.items()):
        for item in items:
            cid = item.get("id")
            if emb_cache is not None and cid in emb_cache:
                emb = emb_cache[cid]
            else:
                emb = embedder.embed_content(item)
                if emb_cache is not None:
                    emb_cache[cid] = emb
            all_embs.append(emb)
            type_indices.append(type_idx)

    if not all_embs:
        return q_values

    # 배치 텐서 생성 및 Q값 추론
    state_t = torch.tensor(state, dtype=torch.float32, device=agent.device)
    state_batch = state_t.unsqueeze(0).repeat(len(all_embs), 1)
    content_batch = torch.tensor(
        np.stack(all_embs), dtype=torch.float32, device=agent.device
    )

    agent.q_net.eval()
    with torch.no_grad():
        q_tensor = agent.q_net(state_batch, content_batch).squeeze(1)
    agent.q_net.train()

    q_list = q_tensor.cpu().numpy().tolist()

    # 타입별로 분배
    type_list = list(cand_dict.keys())
    for idx, qv in enumerate(q_list):
        ctype = type_list[type_indices[idx]]
        q_values[ctype].append(qv)

    return q_values
