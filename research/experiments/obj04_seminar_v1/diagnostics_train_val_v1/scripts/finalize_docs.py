"""이번 작업의 신규 보고서 표현·링크·산출물 무결성을 마무리한다."""
from pathlib import Path
import re,json,csv,hashlib,ast
R=Path(__file__).resolve().parents[1]
p=R/'reports/Train_Val_XYZ_원인분석.md';s=p.read_text()
# Improve spacing of newly generated Korean prose, leaving filenames, code and measurements intact.
replacements={
'GTmask 유효픽셀은 Train261,419, Val64,635이다.':'GTmask 유효 픽셀은 Train 261,419개, Val 64,635개이다.',
'Front Z MAE는Val이Train의':'Front Z MAE는 Val이 Train의',
'Back은':'Back은 ',
'Z0의 GT/Pred=1 비율은대체로50% 부근인데 Val error는약44%이다.':'Z0의 GT/Pred=1 비율은 대체로 50% 부근인데 Val 오류율은 약 44%이다.',
'단일방향 flag는사전 정의한 pooled Pred≤1% 또는≥99% 기준이고, GT 비율도 함께 확인했다.':'한쪽 출력 flag는 사전 정의한 pooled Pred≤1% 또는 ≥99% 기준이며, GT 비율도 함께 확인했다.',
'모든축·plane의 GT/Pred비율·raw mean/std·margin·confidence·TP/TN·balanced accuracy·프레임별 한쪽 출력 수는CSV에 있다.':'모든 축·plane의 GT/Pred 비율, raw mean/std, margin, confidence, TP/TN, balanced accuracy, 프레임별 한쪽 출력 수는 CSV에 있다.',
'첫 측정은기존 진단의':'첫 측정은 기존 진단의',
'정밀도 점검에서FP16 sigmoid가0.5로 round되는 작은 양수는':'정밀도 점검에서 FP16 sigmoid가 0.5로 round되는 작은 양수는',
'원본 dtype/threshold로 재측정한 결과를':'원본 dtype/threshold로 재측정한 결과를',
'현재 가장 강한 판정은 B와 D의 조합':'현재 가장 강한 판정은 B와 D의 조합',
'A의전체적 학습 실패 설명은약하나 fine plane 잔차와 일부Train Pose 실패는 남는다.':'A의 전체적인 학습 실패 설명은 약하지만, fine plane 잔차와 일부 Train Pose 실패는 남는다.',
'Train 블록 Front 평균오차 범위':'Train 블록의 Front 평균 오차 범위는 ',
'인 반면 Val블록4는':'인 반면 Val 블록4는 ',
', 블록9는':', 블록9는 ',
'한두 Val 프레임만의 문제로 설명되지 않는다.':'한두 Val 프레임만의 문제로 설명되지 않는다.',
'블록4 Mask IoU0.9613도':'블록4의 Mask IoU 0.9613도 ',
'같은블록의10프레임은 상관된 표본이므로100개 독립 장면으로 해석하지 않는다.':'같은 블록의 10프레임은 상관된 표본이므로, 100개 독립 장면으로 해석하지 않는다.',
'GT XYZ분포는같은64개 model-axis bin을사용해분할별 합이1이되게 정규화했다.':'GT XYZ 분포는 같은 64개 model-axis bin을 사용해 분할별 합이 1이 되도록 정규화했다.',
'Z0 GT positive가 Train/Val 모두대체로50%여서 단순한클래스불균형으로 coarse gap을 설명하기 어렵다.':'Z0 GT positive 비율은 Train/Val 모두 대체로 50%여서 단순한 클래스 불균형으로 coarse 격차를 설명하기 어렵다.',
'XYZ축별분포·TV·JS 요약은CSV에 보존했고 정식 두분포 검정/인과 판정으로 사용하지 않았다.':'XYZ 축별 분포·TV·JS 요약은 CSV에 보존했고, 정식 두 분포 검정이나 인과 판정으로 사용하지 않았다.',
'GT object-Z와 camera-Z각도 범위는':'GT object-Z와 camera-Z 각도 범위는 ',
'이것만으로 전체6D포즈 분포가같다고 할 수 없다.':'이것만으로 전체 6D Pose 분포가 같다고 할 수 없다.',
'Front XYZ오차와 이각도의Pearson r은 Train-0.0391, Val0.0833이다.':'Front XYZ 오차와 이 각도의 Pearson r은 Train -0.0391, Val 0.0833이다.',
'가시율은모두약0.997~1.0으로 변동이작아 가림 효과를검증하기에부족하다.':'가시율은 모두 약 0.997~1.0으로 변동이 작아 가림 효과를 검증하기에 부족하다.',
'Front XYZ오차와PnP실제inlier비율 r은Train-0.4362, Val-0.3354이며, ADD와는Train0.3292, Val0.7858이다.':'Front XYZ 오차와 PnP 실제 inlier 비율의 r은 Train -0.4362, Val -0.3354이며, ADD와는 Train 0.3292, Val 0.7858이다.',
'같은 생성블록에상관된 기술통계다.':'같은 생성 블록 안에서 상관된 표본의 기술통계다.',
"유효PnP반환 실패는0이었으므로 '실패한PnP'는여기서ADD 기준 미달로구분한다.":"유효 PnP 반환 실패는 0이었으므로, 여기서는 ADD 기준 미달을 별도로 구분한다.",
'ADD통과/실패별 좌표·inlier 집계도추가했다.':'ADD 통과/실패별 좌표·inlier 집계도 추가했다.',
}
for a,b in replacements.items():s=s.replace(a,b)
a=s.index('## 9. 다음 실험에서 한 가지 변경');b=s.index('## 10. 산출물·보존·재현')
s=s[:a]+'''## 9. 다음 실험에서 한 가지 변경

**선택: 학습 crop에 동기화된 in-plane 회전 증강만 추가한다.** 현재 Baseline에 증강이 없고 블록 일반화 격차가 크다는 관찰에 근거한 다음 대조 실험 제안이다. 개선 효과는 미확인이다.

- 고정 정책: Train crop의 회전각 uniform[-20°,20°], seed42. RGB는 bilinear, Mask·XYZ·유효 support는 nearest로 변환하고 HCCE는 변환 후 XYZ에서 생성한다. 모델 좌표값 자체를 회전하지 않고 픽셀 위치만 동기 변환한다.
- 유지할 설정: Train80/Val20 및 블록, random-initialized ResNet34, batch4/crop256, Adam lr0.0002, 최대195epoch/3900step, 기존 Loss 수식·타입 호환, epoch≥1의 Val Loss 체크포인트 선택 규칙. Validation에는 증강을 적용하지 않고 PnP 조건도 유지한다.
- 평가: 기존 Front 주지표와 Z0 bit/XYZ 오차를 함께 보고한다. Train 적합도 변화도 기록하되, 같은 Val을 재사용하므로 독립 테스트 성능이라고 부르지 않는다.
- 사전 확인: CPU에서 RGB·Mask·XYZ에 동일 affine가 적용되는지 검증한다. 증강한 영상에서 PnP를 평가한다면 K도 동일 affine로 변환해야 한다. 원본 GT는 수정하지 않는다.
- pretrained·Loss 가중치·epochs·PnP threshold를 동시에 바꾸지 않는다. 회전 증강은 3D out-of-plane 시점 변화를 완전히 만들지 못하므로, 개선이 없으면 그대로 보고한다.

이 제안은 이번에 실행하지 않았다. 기존 Back/Mixed PnP inlier 집계의 정식 수정은 별도 회귀 작업으로 분리한다. Front-only 주평가와 무관한 수정을 이번 한 변수 비교에 섞지 않는다.

'''+s[b:]
start=s.index('보존대조:')
s=s[:start]+'''보존 대조: 원본 740파일, 이전 세미나 산출물 366파일, Baseline 61파일, 기존 `diagnostics_v1`의 102파일이 모두 기존 manifest의 SHA-256과 일치했다. 체크포인트 해시도 실행 전후 동일했다. 종료 시 GPU compute process는 없었다. 다른 사용자의 compute PID가 보이면 중단하는 guard를 배치 전과 프레임 처리 중간에 적용했다.

이전 `diagnostics_v1/scripts/diagnose.py`와 학습 `scripts/support.py`의 검증된 처리를 재사용했다. 이번 측정 진입점은 `scripts/measure_source_threshold.py`, 그래프·보고서 생성은 `scripts/build_reports.py`이며 실행 원문은 각 `logs/`에 보존했다. 재현 시 새 결과 루트로 절대 경로를 갱신한다. 현재 출력이 있는 경로에 그대로 재실행하지 않는다.

미수행: 재학습, YOLO, 실사27 평가·튜닝, 새로운 seed, Loss gradient 실험, 독립 테스트, 실제 회전 증강 학습. 새로운 모델의 성능 향상은 주장하지 않는다.
'''
p.write_text(s)
# Rewrite the seminar prose for direct presentation use; all values remain the measured ones.
p=R/'reports/세미나_추가진단_요약.md'
s=p.read_text();start=s.index('같은epoch155')
s=s[:start]+f'''같은 epoch155 체크포인트를 batch4·AMP·GT BBox crop으로 비교했다. Train 수치는 학습 데이터 적합도이며 일반화 성능이 아니다. Val의 XYZ·Pose는 이전 진단과 프레임별 최대차이 0으로 재현됐다.

## 슬라이드 1 — Train 적합도와 Val 일반화의 차이

- 핵심 문장: “Front XYZ 오차가 Train 4.21에서 Val 29.40으로 약 6.99배 증가했다.”
- 실제 수치: Mask IoU 0.9740/0.9358, Front ADD 73.75%/5%.
- 그림: [좌표 오차 비교]({R/'train_val_xyz_error.png'}), [Mask와 좌표 오차]({R/'mask_vs_xyz_comparison.png'}).
- 설명: Train에서도 일부 오차가 남는다. 다만 양쪽 모두 전반적으로 학습하지 못한 상태보다, 분리된 블록에서의 일반화 격차가 더 크다.

## 슬라이드 2 — Z coarse plane의 일반화 실패

- 핵심 문장: “Z0 오류가 Train 2.75%에서 Val 44.57%로 급증한다.”
- 실제 수치: Back Z0도 3.57%/43.12%. Val Front Z0만 GT bit로 치환한 reference Z MAE는 16.3542→3.2228이다.
- 그림: [Z plane 비교]({R/'z_plane_error_comparison.png'}).
- 설명: 앞 HCCE plane의 오류는 뒤 binary 자릿값에도 전파된다. Fine plane의 50% 오류와 영향이 같지 않다. GT 치환은 Oracle 진단이며 실제 모델 개선이 아니다.

## 슬라이드 3 — 다음에는 회전 증강 하나만 변경

- 핵심 문장: “다른 설정을 고정하고 Train crop에 동기화된 ±20° 회전 증강만 추가해 비교한다.”
- 실제 수치: Train 8블록의 Front 평균 오차 3.17~6.89, Val 2블록은 27.78/31.03.
- 그림: [블록별 오차]({R/'block_xyz_error.png'}), [GT Z 분포]({R/'gt_z_histogram_comparison.png'}).
- 설명: 개선 효과는 미확인 가설이다. GT label·Mask에 같은 affine를 적용하고 HCCE를 다시 생성해야 한다. 이번 작업에서는 실행하지 않았으며 실사27로 튜닝하지 않는다.

## 발표 시 제한

- Train ADD는 59/80이며 fine Z plane 잔차도 남는다. “XYZ 학습 완료”라고 말하지 않는다.
- 회전 증강은 제안만 했다. 정확도가 개선됐다는 결과는 없다.
- 단일 checkpoint와 Val 2블록의 비교다. 원인이 Loss·초기화·표본량 중 무엇인지 확정하지 않는다.
- FP16 실제 sigmoid 임계값과 `raw>0`의 작은 차이를 분리했다. 현재 CSV는 실제 원본 임계값 기준이며 기존 Pose 수치는 변하지 않았다.

상세: [Train/Val 원인분석]({R/'reports/Train_Val_XYZ_원인분석.md'})
'''
p.write_text(s)
# Compile-only check: no model execution, no bytecode output.
for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
for p in (R/'reports').glob('*.md'):
 for q in re.findall(r'\]\((/[^)]+)\)',p.read_text()):assert Path(q).is_file(),q
required=['train_val_xyz_comparison.csv','train_val_pose_metrics.csv','hcce_plane_error.csv','train_val_xyz_error.png','z_plane_error_comparison.png','mask_vs_xyz_comparison.png','reports/Train_Val_XYZ_원인분석.md','reports/세미나_추가진단_요약.md']
assert all((R/x).is_file() for x in required)
manifest=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(R.rglob('*')) if p.is_file()]
with (R/'checks/final_artifact_manifest.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
print(json.dumps({'required_files':len(required),'verified':True,'new_artifacts':len(manifest),'bytes':sum(x['bytes'] for x in manifest)},ensure_ascii=False))
