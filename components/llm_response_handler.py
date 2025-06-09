import json
import logging
import random
import re
from typing import Any, Dict, List, Tuple


class LLMResponseHandler:
    """
    LLM 시뮬레이터 응답 처리 및 검증을 담당하는 클래스
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Args:
            debug (bool): 디버깅 로그 출력 여부
        """
        self.debug = debug

    def extract_user_response(
        self, llm_raw_text: str, selected_content_id: int, total_contents_count: int
    ) -> Tuple[str, int]:
        """
        LLM 원본 텍스트에서 특정 콘텐츠에 대한 사용자 반응을 추출합니다.

        Args:
            llm_raw_text (str): LLM의 원본 응답 텍스트
            selected_content_id (int): 추출할 콘텐츠 ID
            total_contents_count (int): 전체 콘텐츠 수 (검증용)

        Returns:
            Tuple[str, int]: (이벤트 타입, 체류시간)

        Raises:
            ValueError: 응답 형식이 잘못된 경우
        """
        try:
            # 1. JSON 파싱
            parsed_responses = self._parse_json(llm_raw_text)

            # 2. 응답 수 검증
            self._validate_response_count(parsed_responses, total_contents_count)

            # 3. 특정 콘텐츠 응답 추출
            return self._extract_content_response(parsed_responses, selected_content_id)

        except Exception as e:
            if self.debug:
                logging.error("LLM response processing error: %s", e)
            raise

    def _parse_json(self, text: str) -> List[Dict]:
        """
        LLM 원본 텍스트에서 JSON 배열을 파싱합니다.

        이 메서드는 Markdown 코드 펜스(````json` 또는 ``````)를 제거하고,
        남은 텍스트를 JSON으로 디코딩하여 응답 객체 리스트를 반환합니다.
        지원하는 JSON 구조는 다음 두 가지입니다:
          1. 최상위 요소가 배열인 JSON.
          2. 최상위 요소가 딕셔너리이고, `"responses"` 키에 응답 리스트가 있는 경우.

        Args:
            text (str): LLM에서 생성된 원본 텍스트. Markdown 코드 펜스로 감싸져 있을 수 있습니다.

        Returns:
            List[Dict]: 파싱된 응답 객체들의 리스트.

        Raises:
            ValueError: JSON 구조가 예상과 다르거나 디코딩에 실패했을 때 발생합니다.
        """

        if self.debug:
            logging.debug("JSON 파싱 시작...")

        # JSON 블록 마크다운 제거
        # 기존 text: 마크다운 코드펜스 포함 가능
        text = text.strip()

        # 정규식으로 앞뒤 마크다운 코드펜스 제거
        text = re.sub(r"^```(?:json)?\s*", "", text)  # 앞쪽 ``` 또는 ```json
        text = re.sub(r"\s*```$", "", text)  # 뒤쪽 ```

        # JSON 파싱
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                if self.debug:
                    logging.debug("JSON 파싱 성공: %d개 객체", len(parsed))
                return parsed
            elif isinstance(parsed, dict) and "responses" in parsed:
                responses = parsed["responses"]
                if self.debug:
                    logging.debug(
                        logging.debug(
                            "'responses' 키에서 %d개 객체 파싱", len(responses)
                        )
                    )
                return responses
            else:
                raise ValueError(f"Unexpected JSON structure: {type(parsed)}")
        except json.JSONDecodeError as e:
            if self.debug:
                logging.error("JSON 파싱 실패: %s", e)
            raise ValueError(f"LLM이 올바른 JSON을 생성하지 못했습니다: {text}")

    # def _validate_response_structure(self, response: Dict[str, Any]) -> None:
    #     """LLM 응답의 기본 구조를 검증합니다."""
    #     if not isinstance(response, dict):
    #         raise ValueError(f"LLM response is not a dictionary: {type(response)}")

    #     if "responses" not in response:
    #         raise ValueError(
    #             f"LLM response missing 'responses' key: {list(response.keys())}"
    #         )

    #     if not isinstance(response["responses"], list):
    #         raise ValueError(
    #             f"'responses' is not a list: {type(response['responses'])}"
    #         )

    #     if self.debug:
    #         logging.debug(f"✅ LLM response structure validation passed")

    def _validate_response_count(
        self, responses: List[Dict], expected_count: int
    ) -> None:
        """응답 수가 예상과 일치하는지 검증합니다.

        Args:
            responses (List[Dict]): LLM 또는 시뮬레이터에서 받은 응답 리스트.
            expected_count (int): 기대하는 응답 개수(top-k 등).

        Raises:
            ValueError: 실제 응답 수가 기대 수와 일치하지 않을 경우 발생합니다.
        """
        actual_count = len(responses)

        if actual_count != expected_count:
            raise ValueError(
                "Response count mismatch: expected %d, got %d"
                % (expected_count, actual_count)
            )

        if self.debug:
            logging.debug(
                "Response count validation passed: %d responses", actual_count
            )

    def _extract_content_response(
        self, responses: List[Dict], target_content_id: int
    ) -> Tuple[str, int]:
        """특정 콘텐츠에 대한 응답을 추출합니다.

        Args:
            responses (List[Dict]): 응답 객체들의 리스트.
            target_content_id (int): 추출 대상이 되는 콘텐츠의 ID.

        Returns:
            Tuple[str, int]: 파싱된 응답 결과(포맷 및 값).

        Raises:
            ValueError: 해당 콘텐츠에 대한 응답이 없거나, 응답 포맷이 잘못된 경우.
        """
        for i, resp in enumerate(responses):
            if not isinstance(resp, dict):
                if self.debug:
                    logging.warning("Invalid response format at index %d: %s", i, resp)
                continue

            content_id = int(resp.get("content_id"))
            if content_id == target_content_id:
                return self._parse_single_response(resp, target_content_id)

        # 해당 콘텐츠에 대한 응답을 찾지 못한 경우
        raise ValueError("No response found for content_id: %d" % target_content_id)

    def _parse_single_response(
        self, response: Dict, content_id: int
    ) -> Tuple[str, int]:
        """단일 응답을 파싱하고 검증합니다.

        Args:
            response (Dict): 단일 콘텐츠에 대한 응답 딕셔너리.
            content_id (int): 응답을 검증할 콘텐츠 ID.

        Returns:
            Tuple[str, int]: (이벤트 타입, 체류 시간(초)). 예시: ("CLICK", 123)

        Notes:
            - 클릭하지 않은 경우(event_type="VIEW") 체류시간은 항상 0입니다.
            - 입력 값에 이상이 있으면 경고 로그를 남기고 보정합니다.
        """
        # 클릭 여부 추출 및 검증
        clicked = response.get("clicked", False)
        if not isinstance(clicked, bool):
            if self.debug:
                logging.warning(
                    "Invalid clicked value for %d: %s, using False", content_id, clicked
                )
            clicked = False

        # 체류시간 추출 및 검증
        dwell_time = response.get("dwell_time_seconds", 0)
        if not isinstance(dwell_time, (int, float)) or dwell_time < 0:
            if self.debug:
                logging.warning(
                    "Invalid dwell_time for %d: %s, using 0", content_id, dwell_time
                )
            dwell_time = 0

        # 클릭했는데 체류시간이 0인 경우 로직 검증
        if clicked and dwell_time == 0:
            if self.debug:
                logging.warning("Content %d: clicked=True but dwell_time=0", content_id)

        # 클릭하지 않았는데 체류시간이 있는 경우 0으로 보정
        if not clicked and dwell_time > 0:
            if self.debug:
                logging.warning(
                    "Content %d: clicked=False but dwell_time=%s, correcting to 0",
                    content_id,
                    dwell_time,
                )
            dwell_time = 0

        event_type = "CLICK" if clicked else "VIEW"

        if self.debug:
            logging.debug(
                "Parsed response for %d: %s, %ds", content_id, event_type, dwell_time
            )

        return event_type, int(dwell_time)

    def create_fallback_response(
        self, use_click_probability: float = 0.2
    ) -> Tuple[str, int]:
        """
        LLM 응답 실패 시 사용할 폴백 응답을 생성합니다.

        Args:
            use_click_probability (float): 클릭 확률

        Returns:
            Tuple[str, int]: (이벤트 타입, 체류시간)
        """
        event_type = "CLICK" if random.random() < use_click_probability else "VIEW"
        dwell_time = (
            random.randint(60, 600) if event_type == "CLICK" else random.randint(5, 300)
        )

        if self.debug:
            logging.debug(
                "Generated fallback response: %s, %ds", event_type, dwell_time
            )

        return event_type, dwell_time

    def extract_all_responses(
        self, llm_raw_text: str, all_contents: List[Dict]
    ) -> List[Dict]:
        """
        LLM 원본 텍스트에서 모든 콘텐츠에 대한 사용자 반응을 추출합니다.

        Args:
            llm_raw_text (str): LLM의 원본 응답 텍스트
            all_contents (List[Dict]): 전체 콘텐츠 리스트

        Returns:
            List[Dict]: 각 콘텐츠별 반응 정보
                       [{"content_id": int, "clicked": bool, "dwell_time": int}, ...]
        """
        try:
            # 1. JSON 파싱
            parsed_responses = self._parse_json(llm_raw_text)

            # 2. 응답 수 검증
            self._validate_response_count(parsed_responses, len(all_contents))

            # 3. 모든 콘텐츠 응답 추출 및 변환
            return self._extract_all_content_responses(parsed_responses, all_contents)

        except Exception as e:
            if self.debug:
                logging.error(f"LLM response processing error: {e}")
            # 폴백으로 모든 콘텐츠에 대해 랜덤 응답 생성
            return self._create_fallback_all_responses(all_contents)

    def _extract_all_content_responses(
        self, responses: List[Dict], all_contents: List[Dict]
    ) -> List[Dict]:
        """
        모든 콘텐츠에 대한 응답을 추출하고 변환합니다.

        Args:
            responses (List[Dict]): LLM 또는 시뮬레이터에서 받은 응답 리스트.
            all_contents (List[Dict]): 추천한 전체 콘텐츠 리스트.

        Returns:
            List[Dict]: 각 콘텐츠별 응답 딕셔너리 리스트. (content_id, clicked, dwell_time)
        """

        result = []
        content_ids = [content.get("id") for content in all_contents]

        for i, resp in enumerate(responses):
            if not isinstance(resp, dict):
                if self.debug:
                    logging.warning("Invalid response format at index %d: %s", i, resp)
                # 폴백으로 해당 콘텐츠에 대해 기본 응답 추가
                if i < len(content_ids):
                    result.append(
                        {
                            "content_id": content_ids[i],
                            "clicked": False,
                            "dwell_time": 0,
                        }
                    )
                continue

            content_id = int(resp.get("content_id"))
            if not content_id:
                if self.debug:
                    logging.warning("Missing content_id in response %d: %s", i, resp)
                # 순서대로 매칭 시도
                if i < len(content_ids):
                    content_id = content_ids[i]
                else:
                    continue

            # 단일 응답 파싱
            clicked, dwell_time = self._parse_single_response_for_all(resp, content_id)

            result.append(
                {"content_id": content_id, "clicked": clicked, "dwell_time": dwell_time}
            )

        # 누락된 콘텐츠에 대해 기본 응답 추가
        response_content_ids = {int(resp["content_id"]) for resp in result}
        for content in all_contents:
            content_id = content.get("id")
            if content_id not in response_content_ids:
                if self.debug:
                    logging.warning(
                        "No response for content_id: %s, adding default", content_id
                    )
                result.append(
                    {"content_id": content_id, "clicked": False, "dwell_time": 0}
                )

        if self.debug:
            clicked_count = sum(1 for resp in result if resp["clicked"])
            logging.debug(
                "Extracted %d responses, %d clicked", len(result), clicked_count
            )

        return result

    def _parse_single_response_for_all(
        self, response: Dict, content_id: int
    ) -> Tuple[bool, int]:
        """
        단일 응답을 파싱하여 클릭 여부와 체류시간을 반환합니다.

        Args:
            response (Dict): 단일 콘텐츠에 대한 응답 딕셔너리.
            content_id (int): 검증 대상 콘텐츠 ID.

        Returns:
            Tuple[bool, int]: (클릭 여부, 체류 시간(초)). 예시: (True, 127)

        Notes:
            - 입력 값에 이상이 있으면 경고 로그를 남기고 보정합니다.
            - dwell_time은 음수일 수 없으며, 클릭하지 않은 경우 항상 0입니다.
        """
        # 클릭 여부 추출 및 검증
        clicked = response.get("clicked", False)
        if not isinstance(clicked, bool):
            if self.debug:
                logging.warning(
                    "Invalid clicked value for %d: %s, using False", content_id, clicked
                )
            clicked = False

        # 체류시간 추출 및 검증
        dwell_time = response.get("dwell_time_seconds", response.get("dwell_time", 0))
        if not isinstance(dwell_time, (int, float)) or dwell_time < 0:
            if self.debug:
                logging.warning(
                    "Invalid dwell_time for %d: %s, using 0", content_id, dwell_time
                )
            dwell_time = 0

        # 클릭했는데 체류시간이 0인 경우 로직 검증
        if clicked and dwell_time == 0:
            if self.debug:
                logging.warning("Content %d: clicked=True but dwell_time=0", content_id)

        # 클릭하지 않았는데 체류시간이 있는 경우 0으로 보정
        if not clicked and dwell_time > 0:
            if self.debug:
                logging.warning(
                    "Content %d: clicked=False but dwell_time=%s, correcting to 0",
                    content_id,
                    dwell_time,
                )
            dwell_time = 0

        return clicked, int(dwell_time)

    def _create_fallback_all_responses(self, all_contents: List[Dict]) -> List[Dict]:
        """
        LLM 응답 실패 시 모든 콘텐츠에 대한 폴백 응답을 생성합니다.

        Args:
            all_contents (List[Dict]): 전체 추천 콘텐츠 딕셔너리 리스트.

        Returns:
            List[Dict]: 각 콘텐츠별 폴백 응답 리스트.
                각 딕셔너리는 아래 필드를 포함합니다:
                    - content_id (Any): 콘텐츠의 ID
                    - clicked (bool): 클릭 여부 (30% 확률)
                    - dwell_time (int): 체류 시간(초). 클릭 시 60~300, 미클릭 시 0
        """
        responses = []
        for content in all_contents:
            # 30% 확률로 클릭
            clicked = random.random() < 0.3
            dwell_time = random.randint(60, 300) if clicked else 0

            responses.append(
                {
                    "content_id": content.get("id"),
                    "clicked": clicked,
                    "dwell_time": dwell_time,
                }
            )

        if self.debug:
            clicked_count = sum(1 for resp in responses if resp["clicked"])
            logging.debug(
                "Generated fallback responses: %d total, %d clicked",
                len(responses),
                clicked_count,
            )

        return responses
