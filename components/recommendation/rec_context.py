class RecContextManager:
    """
    추천 시뮬레이션 컨텍스트 관리 클래스.

    cold start 에피소드 동안은 무조건 균등 quota를,
    이후에는 user preference 기반 quota를 사용하도록 제어.
    """

    def __init__(self, cold_start_episodes: int = 10) -> None:
        """
        Args:
            cold_start_episodes (int): cold start 구간에서 고정 quota를 쓸 에피소드 수
        """
        self.total_steps = 0
        self.cold_start = cold_start_episodes

    def reset(self) -> None:
        """컨텍스트를 리셋(에피소드 시작 시 호출)"""
        self.total_steps = 0

    def use_fixed_quota(self) -> bool:
        """
        cold start 구간인지 판별.

        Returns:
            bool: cold start quota 사용 여부
        """
        return self.total_steps < self.cold_start

    def step(self) -> None:
        """step 카운터 증가(에피소드 내 매 step마다 호출)"""
        self.total_steps += 1


def get_recommendation_quota(
    user_pref: dict[str, float],
    context: RecContextManager,
    max_total: int = 6,
    min_per_type: int = 1,
) -> dict[str, int]:
    """
    추천 quota(콘텐츠 타입별 추천 개수) 계산 함수.

    Args:
        user_pref (dict[str, float]): 각 콘텐츠 타입별 선호도 (0~1, 합계=1이 아님)
        context (RecContextManager): 추천 컨텍스트(콜드스타트 등)
        max_total (int): 한 스텝에서 추천할 총 콘텐츠 수
        min_per_type (int): 각 타입별 최소 추천 개수

    Returns:
        dict[str, int]: {콘텐츠 타입: quota(추천 개수)}
    """
    types = list(user_pref.keys())
    n_types = len(types)

    # 예외 방어: 타입이 없거나, max_total < 타입 수 등
    if n_types == 0:
        raise ValueError("No content types provided in user_pref")
    if max_total < min_per_type * n_types:
        raise ValueError(
            f"max_total({max_total}) < min_per_type({min_per_type}) * n_types({n_types})"
        )

    # 콜드스타트: 균등 할당
    if context.use_fixed_quota():
        eq = max_total // n_types
        quota = {t: eq for t in types}
        # 잔여 quota 분배 (예: 6개, 4타입이면 1,1,1,1 → 2개 남음 → 랜덤/고정 분배)
        remain = max_total - eq * n_types
        for t in types[:remain]:
            quota[t] += 1
        return quota

    # 기본: user_pref 비율로 quota 분배
    raw = {t: max(round(user_pref.get(t, 0) * max_total), min_per_type) for t in types}

    # 총합이 max_total과 맞지 않으면 보정 (round, min_per_type 등으로 오차 발생)
    while sum(raw.values()) > max_total:
        # 가장 quota가 초과된 타입에서 -1씩
        diff = {t: raw[t] - user_pref.get(t, 0) * max_total for t in types}
        t_max = max(diff, key=diff.get)
        if raw[t_max] > min_per_type:
            raw[t_max] -= 1
        else:
            # 모두 최소치에 도달하면 강제 break
            break

    while sum(raw.values()) < max_total:
        # 가장 선호도가 높은 타입에 +1
        t_max = max(user_pref, key=user_pref.get)
        raw[t_max] += 1

    return raw
