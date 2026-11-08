# 다음 실험 우선순위

1. **독립 세션 확보:** 별도 날짜/배치/배경/로봇 동작을 포함하고 촬영 원본 K,D 및 pose 출처를 기록한다. 촬영 단위 holdout을 잠근다. 개발27과 BB100은 검증된 독립 시험셋으로 쓰지 않는다.
2. **카메라·pose 검증:** BB D와 실제 calibration 출처, 보정 newK, 로봇 시간 대응 및 물리 단위를 확인한다. 대표 프레임을 별도 측정 또는 재투영 기준으로 검수한다.
3. **RGB detector:** 검증된 독립 촬영군에서 PBR/학습용 CAD amodal bbox 및 유효한 annotation을 사용해 새 detector를 실험한다. 평가는 RGB 입력만으로 하고 미검출도 분모에 넣는다. 검출 confidence별 recall과 pose 성능을 측정한다.
4. **공식 PBR canonical 라벨 검증:** 공식 `pnp_solve_re_4p`와 `modified_sym_Rt` 및 공식 렌더링 순서·HCCE crop/encoding을 별도 생성 코드에서 재현하고 샘플 단위 픽셀/행렬을 대조한다. 오류·중복·평가 그룹이 해결되기 전 추가 PBR 학습은 NO-GO.
5. **실사 약지도:** 자동 가림 신뢰도와 unknown gate를 소수의 독립 기준 데이터에 검증한다. SAM 후보 또는 CAD amodal을 GT로 가정하지 않고 gate가 0이면 실사 supervised loss를 적용하지 않는다. Pose-consistency는 보정·가림·대칭 안정성이 검증될 때만 별도 파일럿으로 진행한다.
6. **실시간/VLM:** detector가 실제 검출하는 전체 경로에 대해 카메라 입력부터 undistortion, detector, pose, PnP, 통신까지 측정한다. Pose 신뢰도·실패 신호와 좌표계/시각을 명시한 인터페이스를 먼저 정의하고 그 다음 VLM 연동을 검토한다.
