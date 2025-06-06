# Moard-Model: 강화학습 기반 추천 시스템

Moard-Model은 강화학습을 활용한 추천 시스템 구현 프로젝트입니다. 이 프로젝트는 사용자-아이템 상호작용을 강화학습 환경으로 모델링하고, DQN(Deep Q-Network) 기반의 에이전트를 통해 최적의 추천 전략을 학습합니다.

## 개발 환경

- Python 3.8 이상
- PyTorch 2.7.0
- Gymnasium 1.1.1
- NumPy 2.2.6
- Pandas 2.2.3
- 기타 의존성 패키지 (requirements.txt 참조)

## 프로젝트 구조

```
moard-model/
├── components/                 # 핵심 컴포넌트 구현
│   ├── agents.py              # DQN 에이전트 구현
│   ├── envs.py                # 추천 환경 구현
│   ├── embedders.py           # 임베딩 모듈
│   ├── rec_utils.py # 후처리 로직 및 Q값 계산 도우미
│   ├── rec_context.py         # 추천 컨텍스트 관리
│   ├── registry.py            # 컴포넌트 레지스트리
│   ├── rewards.py             # 보상 함수 구현
│   ├── base.py                # 기본 클래스 정의
│   ├── candidates.py          # 후보 생성 로직
│   ├── db_utils.py            # 데이터베이스 유틸리티
│   └── __init__.py
├── models/                    # 신경망 모델 구현
│   ├── q_network.py           # Q-Network 구현
│   ├── doc2vec.py             # doc2vec 모델 학습 후 생성
│   └── __pycache__/
├── runner/                    # 실험 실행 관리
│   ├── experiment_runner.py   # 실험 실행 로직
│   └── __init__.py
├── config/                    # 설정 파일
│   └── experiment.yaml        # 실험 설정
├── replay/                    # 경험 리플레이 저장
├── .venv/                     # 가상환경
├── .git/                      # Git 저장소
├── .gitignore                 # Git 무시 파일 목록
├── main.py                    # 메인 실행 파일
├── requirements.txt           # 의존성 패키지 목록
├── sample_recsys.db           # 샘플 데이터베이스
└── README.md                  # 프로젝트 문서
```

## 주요 기능

- QueryAwareCandidateGenerator: 쿼리(keyword) 기반 후보 생성
- DQN 에이전트를 통한 최적 추천 전략 학습
- 추천 리스트 다양성 보장(타입별 최소 1개) 후처리 로직
- 보상 함수를 통한 추천 품질 평가 (클릭 + 체류시간 기반 가중 보상)
- 콜드 스타트 상황 처리
- 최대 스텝 수 제한

## 설치 방법

1. Python 3.8 이상 버전이 필요합니다.
2. 가상환경 생성 및 활성화:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # 또는
   .venv\Scripts\activate  # Windows
   ```
3. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

## 사용 방법

1. 실험 설정:
   - `config/experiment.yaml` 파일에서 실험 파라미터 조정
   - 주요 설정:
     - 환경 설정:
       - cold_start: 콜드 스타트 사용자 수
       - max_steps: 최대 스텝 수
       - top_k: 추천 아이템 수
     - 에이전트 설정:
       - lr: 학습률
       - batch_size: 배치 크기
       - eps_start: 초기 탐험률
       - eps_min: 최소 탐험률
       - eps_decay: 탐험률 감소율
       - gamma: 할인율
       - update_freq: 타겟 네트워크 업데이트 주기
     - 임베더 설정:
       - user_dim: 사용자 임베딩 차원
       - content_dim: 콘텐츠 임베딩 차원
     - 실험 설정:
       - total_episodes: 총 에피소드 수
       - max_recommendations: 최대 추천 수
       - seeds: 실험 시드 값

2. 실험 실행:
   ```bash
   python main.py
   ```

## 주요 컴포넌트

### 환경 (Environment)
- 추천 시스템을 강화학습 환경으로 모델링
- 사용자-아이템 상호작용 시뮬레이션
- 보상 함수를 통한 추천 품질 평가
- 콜드 스타트 상황 처리
- 최대 스텝 수 제한

### 에이전트 (Agent)
- DQN 기반 추천 에이전트
- ε-greedy 탐험 전략
- 경험 리플레이를 통한 안정적인 학습
- 타겟 네트워크를 통한 학습 안정화
- 배치 학습 지원

### 임베더 (Embedder)
- 사용자와 아이템의 특성을 벡터로 변환
- 단순 연결 방식의 임베딩 구현
- 사용자/콘텐츠 차원 설정 가능

## 데이터베이스

- `sample_recsys.db`: 샘플 추천 시스템 데이터베이스
  - 사용자-아이템 상호작용 데이터
  - 사용자/아이템 메타데이터
  - 추천 결과 저장

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
