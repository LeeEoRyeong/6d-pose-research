# BB100 승인 마스크·Pose 스냅샷 (2026-11-11)

100개 ACCEPTED 작업용 마스크(0 배경, 128 ignore, 255 visible)와 Pose NPY, scene GT/K, 154개 저장 이력의 메타데이터 및 마스크를 보존한다. 이력의 RGB 포함 overlay는 제외한다. manifest.csv는 원본과 스냅샷의 SHA-256을 기록한다. RGB, CAD, 모델 가중치, 자동 후보, 렌더, 원본 데이터셋 및 시각화는 포함하지 않으며 별도 백업이 필요하다. 경로는 원래 서버의 참조이며 클론 후 그대로 접근 가능한 상대경로가 아니다. 모든 128 픽셀은 valid mask로 제외해야 한다.

Phase 16A-5 무결성은 미저장 브라우저 편집 확인 전 CONDITIONAL이다. 실제 학습은 배치 분할, 대칭·타깃과 ignore 처리 검증 전 BLOCKED다.
