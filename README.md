# HCCEPose 6D Pose 연구 백업

실사 BB 객체 4의 Front/Back XYZ 및 visible mask, PBR 학습 및 평가를 검토한 연구 코드와 보고서다. 원본 프로젝트와 개인 데이터는 별도 보관한다. 이 저장소에는 실사·PBR 이미지, 원본 pose, CAD 모델, 마스크, 체크포인트 및 가상환경을 포함하지 않는다.

## 구조

- `research/local/`: 원본 작업공간에서 작성한 분석 보고서와 BB 사전검증 코드.
- `research/experiments/obj04_seminar_v1/`: PBR baseline, 진단, 평가 및 GT-mask loss 실험 코드·설정·선택 지표.
- `research/experiments/real_bb100_20261108/`: BB pose 변환과 투영 검증 도구의 코드만 포함.
- `provenance/`: 원본 버전과 백업 정제 내역.
- `LICENSES/`: 원본 프로젝트의 라이선스.

## 사용 시 주의

연구 기록은 기존 환경의 실행 상태를 보존하기 위한 것이다. 데이터와 모델이 없으므로 이 저장소만으로 학습·평가를 실행할 수 없다. 소스 및 설정의 `/RESEARCH_DATA/...`, Windows `C:\RESEARCH_DATA\...`는 비식별화된 자리표시자이며, 사용자가 소유한 외부 데이터 경로와 일치하도록 별도 설정해야 한다. 원본 프로젝트 의존 코드 및 Python 환경도 필요하다. 일부 보고서의 이미지·원본 데이터 링크는 의도적으로 백업에서 빠져 있다. 스크립트를 실행하기 전에 입력과 출력 경로를 확인해 기존 파일이 덮어쓰이지 않게 한다.

BB 연구는 visible mask 독립 검증 전 단계다. CAD 투영 mask를 검증된 visible mask로 간주하지 않는다. 평가 결과의 실사 GT 및 oracle ROI 제한은 각 보고서를 참고한다.
