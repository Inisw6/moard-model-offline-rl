import logging
import random
import json
from typing import Dict, List, Optional, Tuple, Any


class LLMResponseHandler:
    """
    LLM 시뮬레이터 응답 처리 및 검증을 담당하는 클래스
    """
    
    def __init__(self, debug: bool = False):
        """
        Args:
            debug (bool): 디버깅 로그 출력 여부
        """
        self.debug = debug
    
    def extract_user_response(
        self, 
        llm_raw_text: str, 
        selected_content_id: int,
        total_contents_count: int
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
                logging.error(f"LLM response processing error: {e}")
            raise
    
    def _parse_json(self, text: str) -> List[Dict]:
        """LLM 원본 텍스트에서 JSON 배열을 파싱합니다."""
        
        if self.debug:
            logging.debug("🔧 JSON 파싱 시작...")
        
        # JSON 블록 마크다운 제거
        text = text.strip()
        if text.startswith('```json'):
            text = text[7:].strip()
        elif text.startswith('```'):
            text = text[3:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()
        
        # JSON 파싱
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                if self.debug:
                    logging.debug(f"✅ JSON 파싱 성공: {len(parsed)}개 응답")
                return parsed
            elif isinstance(parsed, dict) and "responses" in parsed:
                responses = parsed["responses"]
                if self.debug:
                    logging.debug(f"✅ JSON 파싱 성공: {len(responses)}개 응답 (딕셔너리 형태)")
                return responses
            else:
                raise ValueError(f"Unexpected JSON structure: {type(parsed)}")
        except json.JSONDecodeError as e:
            if self.debug:
                logging.error(f"❌ JSON 파싱 실패: {e}")
            raise ValueError(f"LLM이 올바른 JSON을 생성하지 못했습니다: {text}")
    
    def _validate_response_structure(self, response: Dict[str, Any]) -> None:
        """LLM 응답의 기본 구조를 검증합니다."""
        if not isinstance(response, dict):
            raise ValueError(f"LLM response is not a dictionary: {type(response)}")
        
        if "responses" not in response:
            raise ValueError(f"LLM response missing 'responses' key: {list(response.keys())}")
        
        if not isinstance(response["responses"], list):
            raise ValueError(f"'responses' is not a list: {type(response['responses'])}")
        
        if self.debug:
            logging.debug(f"✅ LLM response structure validation passed")
    
    def _validate_response_count(self, responses: List[Dict], expected_count: int) -> None:
        """응답 수가 예상과 일치하는지 검증합니다."""
        actual_count = len(responses)
        
        if actual_count != expected_count:
            raise ValueError(
                f"Response count mismatch: expected {expected_count}, got {actual_count}"
            )
        
        if self.debug:
            logging.debug(f"✅ Response count validation passed: {actual_count} responses")
    
    def _extract_content_response(
        self, 
        responses: List[Dict], 
        target_content_id: int
    ) -> Tuple[str, int]:
        """특정 콘텐츠에 대한 응답을 추출합니다."""
        
        for i, resp in enumerate(responses):
            if not isinstance(resp, dict):
                if self.debug:
                    logging.warning(f"Invalid response format at index {i}: {resp}")
                continue
            
            content_id = int(resp.get("content_id"))
            if content_id == target_content_id:
                return self._parse_single_response(resp, target_content_id)
        
        # 해당 콘텐츠에 대한 응답을 찾지 못한 경우
        raise ValueError(f"No response found for content_id: {target_content_id}")
    
    def _parse_single_response(self, response: Dict, content_id: int) -> Tuple[str, int]:
        """단일 응답을 파싱하고 검증합니다."""
        
        # 클릭 여부 추출 및 검증
        clicked = response.get("clicked", False)
        if not isinstance(clicked, bool):
            if self.debug:
                logging.warning(f"Invalid clicked value for {content_id}: {clicked}, using False")
            clicked = False
        
        # 체류시간 추출 및 검증
        dwell_time = response.get("dwell_time_seconds", 0)
        if not isinstance(dwell_time, (int, float)) or dwell_time < 0:
            if self.debug:
                logging.warning(f"Invalid dwell_time for {content_id}: {dwell_time}, using 0")
            dwell_time = 0
        
        # 클릭했는데 체류시간이 0인 경우 로직 검증
        if clicked and dwell_time == 0:
            if self.debug:
                logging.warning(f"Content {content_id}: clicked=True but dwell_time=0")
        
        # 클릭하지 않았는데 체류시간이 있는 경우 0으로 보정
        if not clicked and dwell_time > 0:
            if self.debug:
                logging.warning(f"Content {content_id}: clicked=False but dwell_time={dwell_time}, correcting to 0")
            dwell_time = 0
        
        event_type = "CLICK" if clicked else "VIEW"
        
        if self.debug:
            logging.debug(f"✅ Parsed response for {content_id}: {event_type}, {dwell_time}s")
        
        return event_type, int(dwell_time)
    
    def create_fallback_response(self, use_click_probability: float = 0.2) -> Tuple[str, int]:
        """
        LLM 응답 실패 시 사용할 폴백 응답을 생성합니다.
        
        Args:
            use_click_probability (float): 클릭 확률
            
        Returns:
            Tuple[str, int]: (이벤트 타입, 체류시간)
        """
        event_type = "CLICK" if random.random() < use_click_probability else "VIEW"
        dwell_time = random.randint(60, 600) if event_type == "CLICK" else random.randint(5, 300)
        
        if self.debug:
            logging.debug(f"🎲 Generated fallback response: {event_type}, {dwell_time}s")
        
        return event_type, dwell_time

    def extract_all_responses(
        self, 
        llm_raw_text: str, 
        all_contents: List[Dict]
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
        self, 
        responses: List[Dict], 
        all_contents: List[Dict]
    ) -> List[Dict]:
        """모든 콘텐츠에 대한 응답을 추출하고 변환합니다."""
        
        result = []
        content_ids = [content.get("id") for content in all_contents]
        
        for i, resp in enumerate(responses):
            if not isinstance(resp, dict):
                if self.debug:
                    logging.warning(f"Invalid response format at index {i}: {resp}")
                # 폴백으로 해당 콘텐츠에 대해 기본 응답 추가
                if i < len(content_ids):
                    result.append({
                        "content_id": content_ids[i],
                        "clicked": False,
                        "dwell_time": 0
                    })
                continue
            
            content_id = int(resp.get("content_id"))
            if not content_id:
                if self.debug:
                    logging.warning(f"Missing content_id in response {i}: {resp}")
                # 순서대로 매칭 시도
                if i < len(content_ids):
                    content_id = content_ids[i]
                else:
                    continue
            
            # 단일 응답 파싱
            clicked, dwell_time = self._parse_single_response_for_all(resp, content_id)
            
            result.append({
                "content_id": content_id,
                "clicked": clicked,
                "dwell_time": dwell_time
            })
        
        # 누락된 콘텐츠에 대해 기본 응답 추가
        response_content_ids = {int(resp["content_id"]) for resp in result}
        for content in all_contents:
            content_id = content.get("id")
            if content_id not in response_content_ids:
                if self.debug:
                    logging.warning(f"No response for content_id: {content_id}, adding default")
                result.append({
                    "content_id": content_id,
                    "clicked": False,
                    "dwell_time": 0
                })
        
        if self.debug:
            clicked_count = sum(1 for resp in result if resp["clicked"])
            logging.debug(f"✅ Extracted {len(result)} responses, {clicked_count} clicked")
        
        return result
    
    def _parse_single_response_for_all(self, response: Dict, content_id: int) -> Tuple[bool, int]:
        """단일 응답을 파싱하여 클릭 여부와 체류시간을 반환합니다."""
        
        # 클릭 여부 추출 및 검증
        clicked = response.get("clicked", False)
        if not isinstance(clicked, bool):
            if self.debug:
                logging.warning(f"Invalid clicked value for {content_id}: {clicked}, using False")
            clicked = False
        
        # 체류시간 추출 및 검증
        dwell_time = response.get("dwell_time_seconds", response.get("dwell_time", 0))
        if not isinstance(dwell_time, (int, float)) or dwell_time < 0:
            if self.debug:
                logging.warning(f"Invalid dwell_time for {content_id}: {dwell_time}, using 0")
            dwell_time = 0
        
        # 클릭했는데 체류시간이 0인 경우 로직 검증
        if clicked and dwell_time == 0:
            if self.debug:
                logging.warning(f"Content {content_id}: clicked=True but dwell_time=0")
        
        # 클릭하지 않았는데 체류시간이 있는 경우 0으로 보정
        if not clicked and dwell_time > 0:
            if self.debug:
                logging.warning(f"Content {content_id}: clicked=False but dwell_time={dwell_time}, correcting to 0")
            dwell_time = 0
        
        return clicked, int(dwell_time)
    
    def _create_fallback_all_responses(self, all_contents: List[Dict]) -> List[Dict]:
        """
        LLM 응답 실패 시 모든 콘텐츠에 대한 폴백 응답을 생성합니다.
        """
        responses = []
        for content in all_contents:
            # 30% 확률로 클릭
            clicked = random.random() < 0.3
            dwell_time = random.randint(60, 300) if clicked else 0
            
            responses.append({
                "content_id": content.get("id"),
                "clicked": clicked,
                "dwell_time": dwell_time
            })
        
        if self.debug:
            clicked_count = sum(1 for resp in responses if resp["clicked"])
            logging.debug(f"🎲 Generated fallback responses: {len(responses)} total, {clicked_count} clicked")
        
        return responses 