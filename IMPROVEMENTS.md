# 코드 개선 사항 정리

## 요약

현재 저장소는 아이디어 검증용 참조 구현으로는 충분히 흥미롭지만, 코드 규모에 비해 구조화와 배포/테스트 체계가 뒤따르지 못하고 있다. 특히 핵심 로직이 하나의 거대 모듈에 과밀하게 모여 있고, 공개 API 및 README가 지나치게 넓은 범위를 한 번에 노출하고 있어 유지보수 비용이 빠르게 커질 수 있는 상태다.

아래 항목들은 코드베이스를 읽고 실제 실행해 본 뒤 우선순위 순으로 정리한 개선 과제들이다.

## 우선순위 높음

### [x] 1. `scalar_field.py` 초기 모듈 분해

- 현재 핵심 구현이 [`src/relativistic_circuit_locality/scalar_field.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/scalar_field.py)에 집중돼 있다.
- 파일 크기가 7,761라인이고, 함수 171개와 클래스 75개가 한 파일에 공존한다.
- 물리 모델, 수치 적분, 격자 해석, 개방계 근사, 상태 토모그래피, 백리액션, 보조 결과 타입이 한 모듈에 섞여 있어 변경 영향 범위를 예측하기 어렵다.

반영 내용:

- trajectory 및 spline 보간 책임을 [`src/relativistic_circuit_locality/geometry.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/geometry.py)로 분리했다.
- `scalar_field.py`는 분리된 geometry 계층을 import 하도록 바꿨다.
- 전체 파일 분해까지는 아니지만, 가장 기초적인 데이터 모델과 보간 책임을 별도 모듈로 떼어 첫 분리 단계를 반영했다.

개선 방향:

- `geometry`, `propagation`, `wavepacket`, `lattice`, `open_system`, `spectral`, `results` 같은 하위 모듈로 분리한다.
- 공용 데이터 타입과 순수 수학 유틸리티를 먼저 분리하고, 그 위에 기능별 API를 얹는 계층 구조로 바꾼다.
- 내부 구현 함수와 외부 공개 함수를 분리해 import 비용과 인지 부하를 줄인다.

기대 효과:

- 변경 단위가 작아져 리뷰와 테스트 범위를 통제하기 쉬워진다.
- 신규 기능 추가 시 기존 기능 회귀 위험을 줄일 수 있다.

### [x] 2. 공개 API 축소 및 안정화

- [`src/relativistic_circuit_locality/__init__.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/__init__.py)에 매우 많은 심볼이 직접 re-export 되어 있다.
- 현재 구조는 "무엇이 핵심 API이고 무엇이 실험적 surrogate 인지" 구분이 약하다.
- 사용자는 import 자동완성만으로도 전체 표면적을 감당해야 하고, 향후 하위 호환성 부담이 커진다.

반영 내용:

- 안정 진입점만 담는 [`src/relativistic_circuit_locality/core.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/core.py)를 추가했다.
- 확장/실험 기능은 [`src/relativistic_circuit_locality/experimental.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/experimental.py) 네임스페이스로 분리했다.
- 패키지 루트 [`src/relativistic_circuit_locality/__init__.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/__init__.py)는 이제 core API만 노출하고 `experimental` 모듈을 명시적으로 제공한다.
- 데모 코드도 core와 experimental import를 구분하도록 갱신했다.

개선 방향:

- `core` API와 `experimental` API를 분리한다.
- `__init__.py`에서는 최소한의 안정된 entrypoint만 노출한다.
- 부가 기능은 `relativistic_circuit_locality.experimental.*` 또는 서브모듈 import로 접근하게 한다.

기대 효과:

- 패키지의 실제 지원 범위가 선명해진다.
- 리팩터링 자유도가 높아진다.

### [x] 3. 테스트/패키징 실행 경로 정리

- 기본 환경에서 `PYTHONPATH=src python -m unittest discover -s tests -v`는 `numpy` 미설치로 실패했다.
- `.venv` 환경에서도 `PYTHONPATH=src` 없이는 모듈 import 가 실패했다.
- 즉, 테스트가 "정상적인 설치 상태"보다 "특정 실행 습관"에 의존한다.

반영 내용:

- 실행 경로를 editable install 기반으로 전환해 `python -m pip install -e .` 이후 별도 `PYTHONPATH` 없이 예제와 테스트를 실행할 수 있게 했다.
- 개발 편의를 위해 넣었던 shim 패키지와 `sitecustomize.py`를 제거해 import 경로를 배포 구조와 일치시켰다.
- [`pyproject.toml`](/workspaces/relativistic_circuit_locality/pyproject.toml)에서 runtime dependency 에 잘못 들어가 있던 `pip`를 제거했다.
- 패키지 루트 import surface 를 검증하는 테스트를 추가했다.

개선 방향:

- editable install(`pip install -e .`) 또는 테스트 러너 설정으로 `src` 레이아웃을 자동 인식하게 만든다.
- CI에서 별도 환경 변수 없이 테스트가 통과하도록 정리한다.
- `pyproject.toml`의 runtime dependency 에 포함된 `pip`는 제거한다. `pip`는 애플리케이션 런타임 의존성이 아니라 개발/설치 도구다.

기대 효과:

- 로컬/CI/사용자 환경 간 실행 편차가 줄어든다.
- 온보딩과 재현성이 개선된다.

## 우선순위 중간

### [x] 4. README 범위 축소 또는 계층화

- [`README.md`](/workspaces/relativistic_circuit_locality/README.md)는 구현 범위를 매우 길게 나열하고 있다.
- 정보량은 많지만 핵심 사용 시나리오, 안정 지원 범위, 실험적 기능의 구분이 희미하다.
- 현재 README는 소개 문서라기보다 기능 덤프에 가까워 탐색 비용이 높다.

반영 내용:

- [`README.md`](/workspaces/relativistic_circuit_locality/README.md)를 핵심 소개, 빠른 시작, 안정 API, 문서 링크 중심으로 축소했다.
- 상세 기능 목록은 [`docs/API_SURFACE.md`](/workspaces/relativistic_circuit_locality/docs/API_SURFACE.md)로 분리했다.
- 구현 가정, 안정/실험 범위, surrogate 수준, 현재 한계는 [`docs/MODEL_SCOPE.md`](/workspaces/relativistic_circuit_locality/docs/MODEL_SCOPE.md)로 정리했다.

개선 방향:

- README 본문에는 "핵심 기능", "빠른 시작", "안정 API", "실험 기능 링크"만 남긴다.
- 확장 기능 목록은 별도 `docs/` 문서로 분리한다.
- 논문 대응 관계와 구현 가정, surrogate 수준, 한계를 표로 요약한다.

기대 효과:

- 첫 진입 사용자가 저장소 목적을 빠르게 파악할 수 있다.
- 문서 유지보수 비용이 줄어든다.

### [x] 5. 데모 코드를 시나리오별로 분리

- [`src/relativistic_circuit_locality/demo.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/demo.py)는 428라인에 걸쳐 매우 많은 기능을 한 번에 import 하고 출력한다.
- 데모가 풍부하다는 장점은 있지만, "최소 예제" 역할과 "기능 전시장" 역할이 충돌한다.

반영 내용:

- 예제 실행을 [`src/relativistic_circuit_locality/examples/`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/examples/) 아래의 시나리오별 모듈로 분리했다.
- 공통 로직은 [`src/relativistic_circuit_locality/examples/_shared.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/examples/_shared.py)로 옮겼다.
- `python -m relativistic_circuit_locality.examples {core|field|research}` 형태의 CLI entrypoint 를 제공한다.
- 호환 wrapper 로 남아 있던 `demo.py`도 제거해 예제 진입점을 하나로 정리했다.

개선 방향:

- `demo_core.py`, `demo_wavepacket.py`, `demo_lattice.py`, `demo_open_system.py`처럼 분리한다.
- 공통 branch 생성 헬퍼는 별도 모듈로 뺀다.
- CLI 엔트리포인트를 제공해 시나리오별 실행이 가능하도록 한다.

기대 효과:

- 예제의 학습 순서가 자연스러워진다.
- 회귀 테스트용 샘플과 사용자용 예제를 분리하기 쉬워진다.

### [x] 6. 테스트 파일 분할 및 범주화

- 현재 테스트가 [`tests/test_scalar_field.py`](/workspaces/relativistic_circuit_locality/tests/test_scalar_field.py) 한 파일에 2,203라인으로 몰려 있다.
- 실제로 160개 테스트가 잘 통과하지만, 실패 시 탐색성과 책임 경계가 약하다.

반영 내용:

- 공통 import/헬퍼는 [`tests/_shared.py`](/workspaces/relativistic_circuit_locality/tests/_shared.py)로 분리했다.
- 테스트를 [`tests/test_core_locality.py`](/workspaces/relativistic_circuit_locality/tests/test_core_locality.py), [`tests/test_numerical_algorithms.py`](/workspaces/relativistic_circuit_locality/tests/test_numerical_algorithms.py), [`tests/test_physical_fidelity.py`](/workspaces/relativistic_circuit_locality/tests/test_physical_fidelity.py), [`tests/test_entanglement_diagnostics.py`](/workspaces/relativistic_circuit_locality/tests/test_entanglement_diagnostics.py), [`tests/test_pde_lattice.py`](/workspaces/relativistic_circuit_locality/tests/test_pde_lattice.py), [`tests/test_fundamental_limitations.py`](/workspaces/relativistic_circuit_locality/tests/test_fundamental_limitations.py), [`tests/test_extended_limitations.py`](/workspaces/relativistic_circuit_locality/tests/test_extended_limitations.py)로 분할했다.
- 기존 단일 파일 테스트는 제거해 실패 시 책임 영역이 바로 드러나도록 바꿨다.

개선 방향:

- `test_geometry.py`, `test_phase_matrix.py`, `test_lattice.py`, `test_open_system.py` 등으로 분리한다.
- 빠른 단위 테스트와 무거운 통합 테스트를 구분한다.
- CI에서 최소 smoke set 과 전체 회귀 세트를 분리한다.

기대 효과:

- 실패 원인 추적이 빨라진다.
- 특정 영역 리팩터링 시 관련 테스트만 빠르게 돌릴 수 있다.

## 우선순위 낮음

### [x] 7. 정적 검사와 스타일 자동화 추가

- 현재 `pyproject.toml`에는 포맷터, 린터, 타입체커 설정이 없다.
- 이 정도 규모의 단일 수치 코드에서는 스타일 일관성과 타입 검사가 회귀 방지에 큰 도움이 된다.

반영 내용:

- [`pyproject.toml`](/workspaces/relativistic_circuit_locality/pyproject.toml)에 `dev` extra 로 `ruff`와 `mypy`를 추가했다.
- 기본 lint/type-check 규칙을 `ruff`/`mypy` 설정으로 저장소에 명시했다.
- 설치 기반 개발 흐름에서 `pip install -e .[dev]`로 정적 검사 도구를 함께 설치할 수 있게 했다.

개선 방향:

- `ruff`, `mypy` 또는 `pyright`를 도입한다.
- 최소한 공개 API 함수와 주요 결과 타입에 대한 타입 안정성을 강화한다.
- import 정렬, 미사용 코드, 너무 긴 함수 같은 기본 품질 규칙을 자동화한다.

### [x] 8. 대용량 테이블의 로딩 전략 개선

- [`src/relativistic_circuit_locality/lebedev_tables.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/lebedev_tables.py)는 큰 문자열 테이블을 코드에 직접 포함하고 있다.
- 지금 규모에서는 동작에 큰 문제는 없지만, import 시 메모리와 가독성 측면에서 손해가 있다.

반영 내용:

- Lebedev table 을 코드 내 문자열 상수 대신 [`src/relativistic_circuit_locality/data/lebedev_tables.json`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/data/lebedev_tables.json) 패키지 데이터로 분리했다.
- [`src/relativistic_circuit_locality/lebedev_tables.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/lebedev_tables.py)는 이제 `importlib.resources`와 `lru_cache`를 사용해 필요할 때만 테이블을 로드한다.
- [`pyproject.toml`](/workspaces/relativistic_circuit_locality/pyproject.toml)에 package-data 설정을 추가해 editable install 과 배포 모두에서 같은 데이터 경로를 사용하게 했다.

개선 방향:

- 테이블을 별도 데이터 파일로 분리하고 필요 시 lazy loading 한다.
- 캐시 계층을 두어 반복 파싱 비용을 줄인다.

### [x] 9. 성능 병목 구간 계측 추가

- 수치 적분, 격자 계산, surrogate 계열 함수가 많이 존재하지만 성능 병목이 어디인지 드러나는 프로파일링 훅이 없다.

반영 내용:

- [`src/relativistic_circuit_locality/benchmarking.py`](/workspaces/relativistic_circuit_locality/src/relativistic_circuit_locality/benchmarking.py)에 representative workload benchmark 진입점을 추가했다.
- `branch_phase_matrix`, `wavepacket_phase_matrix`, `finite_difference_kg` 세 경로를 반복 실행해 평균/최소/최대 시간을 출력하도록 했다.
- 경량 프로파일링 훅 `profile_call`과 보고서 포맷 함수 `format_benchmark_report`를 제공해 로컬 병목 확인을 쉽게 했다.

개선 방향:

- 대표 워크로드에 대한 간단한 benchmark 스크립트를 추가한다.
- 느린 경로를 찾아 `numpy` 벡터화 또는 캐시 전략을 적용한다.

## 권장 실행 순서

1. 모듈 분해 계획 수립 및 `scalar_field.py` 책임 분리
2. 공개 API 정리와 `__init__.py` 축소
3. 패키징/테스트 실행 경로 정상화
4. 테스트 파일 분할
5. README와 데모 구조 재작성
6. 정적 검사와 benchmark 추가

## 확인 메모

- 기본 시스템 Python에서는 `numpy` 부재로 테스트가 실패했다.
- 저장소의 `.venv` 기준으로는 `PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v` 실행 시 160개 테스트가 통과했다.
