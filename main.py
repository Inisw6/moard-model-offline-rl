import logging

from runner.experiment_runner import ExperimentRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    """실험 전체 루프를 실행하는 메인 엔트리포인트.

    - ExperimentRunner 클래스 인스턴스를 생성한 후,
      설정 파일(config/experiment.yaml)에 정의된 seed별 실험을 순차적으로 실행한다.
    - 실험 결과 및 학습된 모델 등은 지정 경로에 자동 저장된다.
    """
    ExperimentRunner().run_all()
