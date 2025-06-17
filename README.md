# moard-recsys-engine
## 개요
이 프로젝트는 강화학습을 이용한 추천시스템을 실험하고 Offline Learning을 통해 베이스 모델을 생성하기 위한 프레임워크 입니다.

- **Base 클래스를 구현하고, config 설정을 통해 확장 가능**
- 추상화와 레지스터리, config를 통한 실험 환경 제공
- 다양한 시드에 대한 실험 자동화
- 실험 결과 및 학습된 모델의 자동 저장

## 환경
- Python 3.10
- gymnasium
- CUDA 지원 GPU (선택사항)
- Docker (선택사항)

## 실행 방법

### 1. 도커를 사용하는 경우

1. 도커 이미지 빌드:
```bash
docker build -t moard-recsys-engine .
```

2. 도커 컨테이너 실행:
```bash
docker run --gpus all moard-recsys-engine
```

### 2. 로컬 환경에서 실행하는 경우

1. 가상환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows
```

2. 의존성 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 실험환경 설정:
```bash
base.py를 기반하여 새로운 클래스 작성
experiment.yaml 설정
```

4. 실험 실행:
```bash
python main.py
```

## 프로젝트 구조
```
moard-recsys-engine/
├── models/                         # 모델 관련 코드
│   ├── doc2vec.model               # Doc2Vec 모델 파일
│   ├── q_network.py                # 기본 Q-Network 구현
│   ├── dueling_q_network.py        # Dueling Q-Network 구현
│   └── doc2vec.py                  # Doc2Vec 학습 코드
│
├── components/                     # 시스템 컴포넌트
│   ├── registry.py                 # @register 기반 컴포넌트 자동 등록
│   │
│   ├── agents/                     # 강화학습 에이전트
│   │   ├── dqn_agent.py            # DQN 에이전트 구현
│   │   └── dueling_dqn_agent.py    # Dueling DQN 에이전트 구현
│   │
│   ├── recommendation/             # 추천 시스템 관련
│   │   ├── candidates.py           # 후보 아이템 생성
│   │   └── rec_utils.py            # 추천 유틸리티 함수
│   │
│   ├── reward/                     # 보상 함수
│   │   └── rewards.py              # 다양한 보상 함수 구현
│   │
│   ├── simulation/                 # 시뮬레이션 환경
│   │   ├── llm_simulator.py        # LLM 기반 시뮬레이터
│   │   ├── random_simulator.py     # 랜덤 시뮬레이터
│   │   ├── personas.py             # 페르소나 정의
│   │   ├── llm_response_handler.py # LLM 응답 처리
│   │   └── llm_simu.py             # LLM 시뮬레이션 로직
│   │
│   ├── embedder/                   # 임베딩 관련
│   │   ├── user_weighted.py        # 사용자 가중치 임베딩
│   │   ├── simple.py               # 기본 임베딩
│   │   ├── simple_concat.py        # 단순 연결 임베딩
│   │   ├── content_sbert.py        # SBERT 기반 콘텐츠 임베딩
│   │   └── content_doc2vec.py      # Doc2Vec 기반 콘텐츠 임베딩
│   │
│   ├── database/                   # 데이터베이스 관련
│   │   ├── persona_db.py           # 페르소나 DB 관리
│   │   └── db_utils.py             # DB 유틸리티 함수
│   │
│   └── core/                       # 핵심 컴포넌트
│       ├── base.py                 # 기본 추상 클래스 정의
│       └── envs.py                 # 실험 환경 정의
│
├── data/                           # 데이터 파일
│   ├── logs/                       # 로그 파일
│   ├── models/                     # 학습된 모델 저장
│   ├── sample_recsys.db            # 학습 샘플 DB
│   ├── personas.db                 # 페르소나 DB
│   └── default_personas.json       # 기본 페르소나 정의
│
├── graph/                          # 그래프 관련 
│   ├── epi_graph.py                # 에피소드 그래프
│   └── step_graph.py               # 스탭 그래프
│
├── runner/                         # 실험 실행 관련
│   └── experiment_runner.py        # 실험 실행기
│
├── config/                         # 설정 파일
│   └── experiment.yaml             # 실험 설정
├── main.py                         # 메인 실행 파일
├── requirements.txt                # 의존성 관리
└── Dockerfile                      # 도커 파일
```

## 주요 컴포넌트 설명

### 1. 모델 (models/)
- **q_network.py**  
  기본 DQN Q-Network 아키텍처 정의. 다층 퍼셉트론(MLP)으로 사용자 상태와 콘텐츠 임베딩을 연결해 Q-값을 예측합니다.
- **dueling_q_network.py**  
  듀얼링 DQN 구조 구현. 상태 가치(state value)와 어드밴티지(advantage)를 분리 학습해 안정성을 높입니다.
- **doc2vec.py**  
  Doc2Vec 기반 텍스트 임베딩 학습 스크립트. 학습 후 `doc2vec.model` 파일로 저장·로딩합니다.
- **doc2vec.model**  
  사전 학습된 Doc2Vec 임베딩 바이너리 파일.

### 2. 컴포넌트 (components/)

#### 2.1 agents/
- **dqn_agent.py**  
  ε-greedy 탐색, 경험 리플레이, Q-네트워크 업데이트, 타깃 네트워크 동기화 등을 포함한 DQN 에이전트 전체 파이프라인.
- **dueling_dqn_agent.py**  
  듀얼링 DQN 에이전트 확장 클래스.

#### 2.2 recommendation/
- **candidates.py**  
  쿼리 및 사용자 프로필 기반 후보 아이템 필터링 로직.
- **rec_utils.py**  
  Q-값 정렬, 다양성(diversity) 제약, 슬레이트 추천 구성 함수 등 추천 후처리 유틸리티.

#### 2.3 reward/
- **rewards.py**  
  클릭(+1.0), 조회(+0.1) 기본 보상 외에 체류 시간, 세션 유지, 다양성 보너스를 추가로 정의할 수 있는 확장형 보상 함수.

#### 2.4 simulation/
- **llm_simulator.py**  
  LLM API 호출 프롬프트 생성 및 결과 획득.
- **llm_response_handler.py**  
  LLM 응답을 클릭/뷰 이벤트와 체류시간으로 파싱해 환경 로그로 변환.
- **random_simulator.py**  
  랜덤 행동 기반 간단 시뮬레이터(베이스라인 용).
- **personas.py**  
  다양한 사용자 페르소나(Persona) 클래스 정의.
- **llm_simu.py**  
  LLM 호출부터 로그 엔트리 생성까지 전체 시뮬레이션 워크플로우 통합.

#### 2.5 embedder/
- **simple.py**  
  사용자 로그 통계(평균 비율, 평균 체류시간 등) 기반 기본 임베딩.
- **user_weighted.py**  
  로그 항목별 가중치를 반영해 사용자 임베딩 개선.
- **simple_concat.py**  
  사용자·콘텐츠 임베딩 벡터 단순 연결.
- **content_sbert.py**  
  SBERT 모델 활용한 콘텐츠 텍스트 임베딩.
- **content_doc2vec.py**  
  Doc2Vec 임베딩을 불러와 콘텐츠 표현으로 활용.

#### 2.6 database/
- **persona_db.py**  
  페르소나 정보 CRUD 및 관리 기능.
- **db_utils.py**  
  DB 연결·세션 관리·쿼리 특화 헬퍼 함수.

#### 2.7 core/
- **base.py**  
  `BaseComponent` 추상 클래스와 `@register` 기반 플러그인 레지스트리 정의.
- **envs.py**  
  Gym API 호환의 `RecEnv` 환경 구현. 상태/행동/보상/종료 로직 캡슐화.

### 3. 데이터 (data/)
- **logs/**  
  시뮬레이션 및 실사용 로그(클릭·체류 등) CSV/JSON 저장 폴더.
- **models/**  
  학습된 Q-Network 및 임베딩 모델 체크포인트 저장 폴더.
- **sample_recsys.db**  
  예제 콘텐츠 및 사용자 로그를 담은 SQLite DB.
- **personas.db**  
  페르소나 메타데이터 저장용 SQLite DB.
- **default_personas.json**  
  초기 페르소나 정의 JSON 파일.

### 4. 실행 (runner/)
- **experiment_runner.py**  
  에피소드 반복 실행, 시드 고정, 메트릭 집계, 결과 저장 및 레포트 생성 담당.

## 실험 설정

프로젝트의 모든 하이퍼파라미터, 환경 및 컴포넌트 설정은 `config/experiment.yaml` 에 정의되어 있습니다.  
아래 예시는 주요 블록을 포함한 기본 구성입니다. 필요에 따라 값과 키를 조정하세요.

```yaml
experiment:
  experiment_name: "dqn_base"           # 실험 식별자
  total_episodes: 10                    # 전체 에피소드 수
  max_recommendations: 6                # 에피소드당 추천 아이템 수
  max_stocks: 1                         # 도메인별 최대 추천 수
  seeds: [0]                            # 재현용 시드 목록
  step_log_path: "data/logs/{experiment_name}/seed_{seed}/steps.csv"
  episode_log_path: "data/logs/{experiment_name}/seed_{seed}/episodes.csv"
  model_save_dir: "data/models/{experiment_name}/seed_{seed}"

env:
  type: rec_env                         # 사용할 환경의 registry 키
  params:
    max_steps: 10                       # 에피소드당 최대 스텝 수
    top_k: 6                            # 한 스텝당 추천 후보 수

agent:
  type: dueling_dqn                     # 에이전트 종류(dqn 또는 dueling_dqn)
  params:
    lr: 0.001                           # 학습률
    batch_size: 32                      # 미니배치 크기
    eps_start: 1.0                      # ε 탐색 시작 값
    eps_min: 0.05                       # ε 탐색 최소 값
    eps_decay: 0.995                    # ε 감소 비율
    gamma: 0.99                         # 할인율
    update_freq: 3                      # 타깃 네트워크 동기화 주기
    loss_type: "smooth_l1"              # 손실 함수 종류('mse' 또는 'smooth_l1')

replay:
  capacity: 10000                       # 경험 리플레이 버퍼 크기

embedder:
  type: simple_concat                   # 임베딩 조합 키
  params:
    user_embedder:
      type: weighted_user               # 사용자 임베딩 방식
      params:
        user_dim: 300                   # 사용자 임베딩 차원
    content_embedder:
      type: doc2vec_content             # 콘텐츠 임베딩 방식
      params:
        content_dim: 300                # 콘텐츠 임베딩 차원

candidate_generator:
  type: query                           # 후보 생성기(registry 키)
  params:
    max_count_by_content: 24            # 타입별 최대 후보 개수

response_simulator:
  type: llm                             # 시뮬레이터 종류('random' 또는 'llm')
  params:
    persona_id: 5                       # 사용할 페르소나 ID(null 시 랜덤)
    debug: true                         # 디버그 로깅 여부
    llm_simulator:
      params:
        provider: "openrouter"         # LLM 제공자
        model: "meta-llama/llama-3.3-70b-instruct"
        api_base: "https://openrouter.ai/api/v1"
        api_key: ""                     # API 키
        temperature: 0.7
        top_p: 0.9
        max_tokens: 1000
        timeout: 30
        debug: true

reward_fn:
  type: default                         # 보상 함수 종류('default' 등)
  params: {}                            # 추가 파라미터
