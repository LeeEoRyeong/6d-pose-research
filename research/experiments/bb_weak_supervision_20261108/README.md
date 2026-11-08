# BB 약한 감독 혼합 학습: CPU smoke 전용

현재 상태는 **NO-GO**다. 검증된 실사 자동 가림 추정기나 confidence가 없으므로 `U_front`, `U_back`, `Q+`, `Q-`가 전부 0이다. CAD amodal은 visible GT가 아니다. 샘플 외 100장 라벨 변환, GPU, 본 학습 및 체크포인트 저장 기능은 실행하지 않는다.

## 실행

```bash
cd /RESEARCH_DATA/HCCEPose_experiments/bb_weak_supervision_20261108
/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/env/bin/python -B smoke.py
/RESEARCH_DATA/HCCEPose_experiments/obj04_seminar_v1/env/bin/python -B inventory.py
```

각 실행은 `runs/` 아래 새 고유 폴더를 만들며 기존 결과를 덮어쓰지 않는다. `smoke.py`는 BB 43/75/235/247과 PBR 0/10만 렌더링하고 BB 4장 pseudo NPZ를 저장한다. `inventory.py`는 BB 100장의 JSON과 이미지 경로 메타데이터만 확인하며 렌더링하지 않는다. `training.mixed_step`은 CPU에서만 실행되며 유효 gate가 없으면 중단한다. `allow_empty_for_smoke=True`는 smoke 검증에만 쓴다. GPU 본 학습 CLI, 자동 분할 모델, 신뢰도 calibration, B/C 완전 학습 및 평가 루프는 아직 구현되지 않았다.

- `pipeline.py`: 기존 CPU 렌더러·HCCE·crop, BB 데이터 adapter, 동일 pose 참조 후보와 그룹 균형 sampler.
- `weak_loss.py`: 기존 PBR criterion에 그대로 위임하고 실사 gate가 있는 지점에만 별도 좌표/mask loss를 계산한다. 실사는 PBR history를 갱신하지 않는다.
- `training.py`: 3 PBR + 1 real 단일 step, lambda 0.25 시작값, 빈 real gate 차단 및 CPU 전용 제한.
- `smoke.py`: synthetic gradient/history, PBR 2장·BB 4장 기하학 및 1 CPU step 검사.
- `inventory.py`: 프레임별 선택 결과와 제외 사유. 현재 100장 모두 신뢰 추정기 부재로 제외한다.
- `config.json`: 계획값 요약. 현재 코드 상수와 일치하지만 자동 설정 로더는 아니므로 config 편집만으로 동작이 바뀌지 않는다.

기존 데이터, 원본 코드 및 체크포인트를 변경하지 않는다. 자세한 결과와 한계는 [결과 보고서](결과_보고서.md)를 참조한다.
