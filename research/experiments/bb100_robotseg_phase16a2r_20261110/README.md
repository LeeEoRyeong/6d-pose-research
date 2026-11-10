# Phase 16A-2R

RobotSeg checkpoint는 서버에서 접근되지 않아 추론하지 않았다. `연구_보고서.md`의 차단 사유 및 코드 감사를 확인한다.

`BB100_robotseg_blocked_review_windows.zip`은 이전 SAM/CAD/LiteFlowNet 후보 검토용이다. ZIP을 통째로 풀고 `index.html`을 오프라인 Chrome으로 연다. RobotSeg 결과는 포함돼 있지 않다. 외부에서 직접 보정한 단일 채널 PNG를 불러와 별도의 미검증 참조 파일로 다운로드할 수 있으나 이 도구는 픽셀 편집을 하지 않는다.

`package.py`는 별도 새 디렉터리에서 `template.html`과 함께 실행할 수 있다. 서버의 이전 실험 파일 경로가 필요하며 기존 패키지를 덮어쓰지 않는다. `source_audit/`는 commit 고정된 소량 공개 소스 감사본으로 완전한 설치본이 아니다.
