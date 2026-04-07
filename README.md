# relativistic_circuit_locality

`q-2026-03-24-2046.pdf`
("Circuit locality from relativistic locality in scalar field mediated entanglement", arXiv:2305.05645v4)
논문의 핵심 아이디어를 Python으로 옮긴 참조 구현이다.

이 저장소는 논문 전체를 완전한 양자장론 시뮬레이터로 재현하려는 목적이 아니라, 논문에서 직접 계산 가능한 핵심 구성요소를 작고 의존성 없는 수치 모델로 구현하는 데 초점을 둔다.

구현 범위는 다음과 같다.

- 두 quantum-controlled source의 branch별 궤적 표현
- field mediation circuit 성립 여부를 위한 spacelike separation 판정
- 논문의 `d_min > 0` 조건에 대응하는 최소 접근 거리 계산
- on-shell scalar-field phase `theta_rs = -1/2 ∫ rho_rs phi_rs`의 준정적 Yukawa 근사
- branch 조합 사이의 상대 얽힘 위상 계산
- 유한 개 momentum mode 에 대한 Fourier-space displacement amplitude 와 coherent-state 자유 진화 추적

현재 코드는 논문의 parametric approximation 안에서 동작한다. 즉, 완전한 QFT 동역학을 직접 적분하지 않고, 시간 이산화된 궤적과 준정적 상호작용 커널을 이용해 논문의 구조를 계산 가능한 형태로 단순화했다.

최근 개선으로 branch 들이 동일한 시간 샘플을 공유해야 한다는 제약을 없앴다. 서로 다른 시간 이산화로 주어진 궤적도 겹치는 시간 구간에서 선형 보간하여 최소 접근 거리, mediation interval, branch phase 를 계산한다.

## 파일 구성

- `src/relativistic_circuit_locality/scalar_field.py`: 핵심 모델과 수치 계산 함수
- `src/relativistic_circuit_locality/demo.py`: 최소 실행 예제
- `tests/test_scalar_field.py`: 단위 테스트
- `CHAT.md`: 논문 해석, 구현된 기능, 추가 필요 기능 정리

## 빠른 실행

예제 실행:

```bash
PYTHONPATH=src python -m relativistic_circuit_locality.demo
```

테스트 실행:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
