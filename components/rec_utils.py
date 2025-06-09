from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def enforce_type_constraint(
    q_values: Dict[str, List[float]], top_k: int = 6
) -> List[Tuple[str, int]]:
    """
    타입별 Q값 리스트가 담긴 q_values를 받아,
    상위 Q 순으로 top_k 개를 뽑되, 누락된 타입이 있으면 해당 타입 중
    가장 높은 Q값 아이템을 끼워넣어 최종 (ctype, idx) 리스트를 반환합니다.

    Args:
        q_values: {
            'youtube': [q0, q1, q2, ...],
            'blog':    [q0, q1, q2, ...],
            'news':    [q0, q1, q2, ...],
            ...
        }
        top_k: 최종 추천 개수 (예: 6)

    Returns:
        List of (ctype, idx) 튜플, 길이 == top_k.
        각 튜플은 “해당 타입 리스트 내에서 인덱스 idx”를 가리킵니다.
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
    """
    현재 상태(state)와 후보 딕셔너리(cand_dict)를 받아,
    각 타입별 후보들의 Q값 리스트를 반환합니다.

    Args:
        state: 현재 사용자 임베딩 벡터 (np.ndarray 등)
        cand_dict: {
            'youtube': [content0, content1, ...],
            'blog':    [content0, content1, ...],
            'news':    [content0, content1, ...],
            ...
        }
        embedder: 콘텐츠 임베딩을 생성하는 객체 (embedder.embed_content 메서드 사용)
        agent: Q-network가 포함된 에이전트 (agent.q_net 호출)
        emb_cache: (Optional) 콘텐츠 임베딩 캐시 (content_id → embedding)

    Returns:
        {
            'youtube': [q0, q1, ...],
            'blog':    [q0, q1, ...],
            'news':    [q0, q1, ...],
            ...
        }
    """
    import torch

    q_values: Dict[str, List[float]] = {}
    for ctype, contents in cand_dict.items():
        content_embs = []
        for c in contents:
            # 각 콘텐츠마다 캐시 우선 조회
            cid = getattr(c, "id", id(c))
            if emb_cache is not None and cid in emb_cache:
                content_emb = emb_cache[cid]
            else:
                content_emb = embedder.embed_content(c)
                if emb_cache is not None:
                    emb_cache[cid] = content_emb
            content_embs.append(content_emb)
        if not content_embs:
            q_values[ctype] = []
            continue

        us = torch.FloatTensor(state).unsqueeze(0)
        us_rep = us.repeat(len(content_embs), 1).to(agent.device)
        ce = torch.stack([torch.FloatTensor(e) for e in content_embs]).to(agent.device)
        with torch.no_grad():
            q_out = agent.q_net(us_rep, ce).squeeze(1)
        q_list = q_out.cpu().numpy().tolist()
        q_values[ctype] = q_list

    return q_values
