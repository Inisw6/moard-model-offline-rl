from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def enforce_type_constraint(
    q_values: Dict[str, List[float]], top_k: int = 6
) -> List[Tuple[str, int]]:
    """타입별 Q값 리스트에서 top_k개를 뽑으면서 모든 타입을 최소 1개 이상 포함시켜 반환합니다.

    Q값이 높은 순서로 top_k개를 선택하되, 만약 일부 타입이 아예 누락된다면 해당 타입 내 최상위 Q값 아이템을 강제로 포함시킵니다.
    (선정된 리스트에서 최하위 Q값 항목과 교체)

    Args:
        q_values (Dict[str, List[float]]):
            타입별 Q값 리스트. 예시: {
                'youtube': [q0, q1, ...],
                'blog':    [q0, q1, ...],
                'news':    [q0, q1, ...]
            }
        top_k (int): 최종 반환할 (타입, 인덱스) 쌍의 개수.

    Returns:
        List[Tuple[str, int]]: [(ctype, idx), ...] 형태의 top_k 길이 리스트.
            각 튜플은 (콘텐츠 타입, 해당 타입 내 인덱스)를 의미합니다.
    """
    # 1) 모든 (ctype, idx, q) 튜플을 평탄화
    all_items: List[Tuple[str, int, float]] = []
    for ctype, q_list in q_values.items():
        for idx, qv in enumerate(q_list):
            all_items.append((ctype, idx, qv))

    # 2) Q값 기준으로 내림차순 정렬
    all_items.sort(key=lambda x: x[2], reverse=True)

    # 3) 상위 top_k 개 선택 (임시 리스트)
    selected = all_items[:top_k]

    # 4) 현재 포함된 타입 집합
    included_types = {ctype for ctype, _, _ in selected}

    # 5) 누락된 타입 파악
    missing_types = set(q_values.keys()) - included_types

    # 6) 누락 타입마다 한 개씩 보충
    #    - 각 누락 타입의 Q값 리스트에서 가장 큰 Q값을 가진 아이템을 찾아 selected에 삽입
    #    - 그 자리에 있던 최하위 항목(6번째)과 교체
    #    - 단, 해당 타입 리스트가 비어 있으면 무시
    for mtype in missing_types:
        q_list = q_values.get(mtype, [])
        if not q_list:
            continue
        # 해당 타입 내에서 가장 큰 Q값의 인덱스 찾기
        best_idx = int(np.argmax(q_list))
        best_q = q_list[best_idx]
        # 이미 selected에 포함된 같은 (ctype, idx)인지 확인
        if any(ctype == mtype and idx == best_idx for ctype, idx, _ in selected):
            continue
        # 마지막 원소(최하위)를 제거하고 새 아이템 삽입
        # - 제거할 인덱스: selected[top_k-1]
        selected[-1] = (mtype, best_idx, best_q)
        # 다시 정렬하여 최종 top_k 유지
        selected.sort(key=lambda x: x[2], reverse=True)

    # 7) 최종 결과를 (ctype, idx) 튜플 리스트로 반환
    return [(ctype, idx) for ctype, idx, _ in selected]


def compute_all_q_values(
    state: Any,
    cand_dict: Dict[str, List[Dict[str, Any]]],
    embedder: Any,
    agent: Any,
    emb_cache: Optional[Dict[Any, Any]] = None,
) -> Dict[str, List[float]]:
    """주어진 상태(state)와 후보 콘텐츠 딕셔너리(cand_dict)로 타입별 Q값 리스트를 계산합니다.

    모든 타입의 후보 콘텐츠를 하나의 배치로 묶어 Q-network로 Q값을 일괄 추론합니다.
    emb_cache가 주어지면 임베딩 캐싱을 활용합니다.

    Args:
        state (Any): 현재 상태 벡터 (np.ndarray 등).
        cand_dict (Dict[str, List[Dict[str, Any]]]): 타입별 후보 콘텐츠 딕셔너리.
        embedder (Any): embed_content 메서드를 가진 임베딩 객체.
        agent (Any): q_net (Q-network)을 포함하고, device 속성을 가진 객체.
        emb_cache (Optional[Dict[Any, Any]]): 콘텐츠 id → 임베딩을 저장하는 캐시(선택).

    Returns:
        Dict[str, List[float]]: 타입별 Q값 리스트.
    """
    import torch

    q_values: Dict[str, List[float]] = {ctype: [] for ctype in cand_dict.keys()}
    all_content_embs = []
    type_indices = []  # 각 임베딩이 어떤 타입에 속하는지 기록

    # 1. 모든 후보 콘텐츠의 임베딩을 한 번에 수집
    for type_idx, (ctype, contents) in enumerate(cand_dict.items()):
        if not contents:
            continue
        for c in contents:
            cid = c.get("id")
            if emb_cache is not None and cid in emb_cache:
                content_emb = emb_cache[cid]
            else:
                content_emb = embedder.embed_content(c)
                if emb_cache is not None:
                    emb_cache[cid] = content_emb
            all_content_embs.append(content_emb)
            type_indices.append(type_idx)

    if not all_content_embs:
        return q_values

    # 2. 하나의 배치로 텐서 변환 및 Q-value 일괄 계산
    state_tensor = torch.tensor(np.array(state, dtype=np.float32), device=agent.device)
    state_rep = state_tensor.unsqueeze(0).repeat(len(all_content_embs), 1)

    # numpy 배열 리스트를 stack하여 텐서 생성
    content_embs_tensor = torch.tensor(
        np.array(all_content_embs), dtype=torch.float32, device=agent.device
    )

    # 평가 모드 설정
    agent.q_net.eval()
    with torch.no_grad():
        all_q_out = agent.q_net(state_rep, content_embs_tensor).squeeze(1)
    # 다시 학습 모드로 전환
    agent.q_net.train()

    all_q_list = all_q_out.cpu().numpy().tolist()

    # 3. 계산된 Q-value를 다시 타입별로 분배
    type_map = list(cand_dict.keys())
    for i, q_val in enumerate(all_q_list):
        ctype = type_map[type_indices[i]]
        q_values[ctype].append(q_val)

    return q_values
