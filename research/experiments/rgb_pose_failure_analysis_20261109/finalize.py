from pathlib import Path
import csv,json,collections,numpy as np
from scipy.stats import spearmanr
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent; f=list(csv.DictReader((P/'frame_comparison.csv').open()));rs=list(csv.DictReader((P/'roi_comparison.csv').open()));br=list(csv.DictReader((P/'bb100_failure_analysis.csv').open()));st=json.loads((P/'failure_statistics.json').read_text()); base={r['frame']:r for r in rs if float(r['bbox_scale'])==1.0};best={r['frame']:r for r in rs if float(r['bbox_scale'])==1.2};good=[k for k in base if base[k]['symmetry_ADD_pass']=='False' and best[k]['symmetry_ADD_pass']=='True'];bad=[k for k in base if base[k]['symmetry_ADD_pass']=='True' and best[k]['symmetry_ADD_pass']=='False'];
tool={'57','60','63','67','71','74','75','77','79','81','83','85','87','89','90','99','101','103','105','107','109','336','338','358','359','367','371','388'}
# Records visual observation only, with no occlusion claim. All five contact sheets reviewed.
for r in br:
 if r['symmetry_ADD_pass']=='False':r['visual_review']='contact_sheet_inspected; tool_visible_in_crop' if r['frame'] in tool else 'contact_sheet_inspected; tool_not_seen_in_crop';r['overlay_relative_path']='overlays/bb100_failures/'+r['frame'].zfill(6)+'.jpg'
 else:r['overlay_relative_path']=''
with (P/'bb100_failure_analysis.csv').open('w',newline='',encoding='utf8') as z:w=csv.DictWriter(z,fieldnames=br[0].keys());w.writeheader();w.writerows(br)
vals=[r for r in f if r['bbox_iou'] and r['yolo_symmetry_ADD']];st['bbox_iou_vs_symmetry_ADD_spearman']=float(spearmanr([float(x['bbox_iou']) for x in vals],[float(x['yolo_symmetry_ADD']) for x in vals]).statistic);st['roi_scale_1_2_gained_frames']=good;st['roi_scale_1_2_lost_frames']=bad;st['visual_tool_in_failure_crop']=sum(r['frame'] in tool for r in br if r['symmetry_ADD_pass']=='False');st['visual_tool_observation_limit']='qualitative crop review, tool presence is not confirmed occlusion, original GT overlay can obscure image';(P/'failure_statistics.json').write_text(json.dumps(st,ensure_ascii=False,indent=2)+'\n')
# Convenient paired panels of all four changed cases.
for fid in good+bad:
 images=[]
 for scale in ['1.00','1.20']:
  im=Image.open(P/'overlays/roi'/f'{int(fid):06d}_scale_{scale}.jpg');im.thumbnail((800,450));images.append(im)
 panel=Image.new('RGB',(1600,480),'black');draw=ImageDraw.Draw(panel)
 for j,im in enumerate(images):panel.paste(im,(800*j,30));draw.text((800*j+10,8),('1.00 original','1.20 exploratory')[j],fill='white')
 panel.save(P/'overlays/roi'/f'comparison_{int(fid):06d}.jpg',quality=90)
cat=collections.defaultdict(list)
for r in f:cat[r['category']].append(r['frame'])
roi='\n'.join(f"| {r['bbox_scale']:.2f} | {r['pnp']}/27 | {r['symmetry_ADD']}/27 | {r['ADD_S']}/27 | {r['ADD']}/27 | {'—' if r['xyz_stability_frame_median_mm'] is None else format(r['xyz_stability_frame_median_mm'],'.2f')} |" for r in st['roi_summaries'])
md=f'''# RGB-only Pose 실패 원인 및 ROI 확장 실험

2026-11-09. **개발27에서 YOLO bbox 1.20배 확장은 명시적 대칭 ADD를 17/27에서 19/27로 바꿨다.** 3장이 성공으로, 1장이 실패로 바뀌었고 ADD-S는 23/27에서 21/27로 하락했다. 같은 개발 데이터에서 가장 좋은 배율을 선택한 탐색 결과이므로 독립적인 개선 검증은 아니다. 공식 HCCEPose와 YOLO 가중치는 고정했고 학습하지 않았다.

## 실험 기준과 재현

기존 detector 출력의 bbox·confidence를 재사용해 동일 프레임에서 1.00, 1.05, 1.10, 1.15, 1.20배로 각 가로·세로 길이를 확대했다. 중심은 유지하고 각 bbox에 원래 HCCEPose square pad 1.5, 256 crop, 동일 K·FP16 inference·front XYZ PnP(1.5 px, 150 iteration, seed=42+frame)를 적용했다. 따라서 1.00은 기존 정책, 1.20은 **기존 1.5 pad 전 bbox의 가로·세로를 각각 20% 확장**한 정책이다. 검출 결과와 bbox를 GT로 보정하지 않았다. 임계값은 공식 모델 직경의 0.1=13.9379 CAD mm이며 명시적 대칭은 `T_gt @ S`의 회전·이동을 모두 적용한다.

1.00배 27개 결과는 기존 `pose_eval.csv`의 PnP 여부·명시적 대칭 ADD 성공 여부와 일치했고, 유효 프레임의 명시적 대칭 ADD 차이는 0.001 mm 미만이었다. 공식·YOLO 가중치를 포함한 주요 입력 8개의 SHA-256은 전후 일치했다(전 데이터 전수 검사는 아님). 추론 실행 코드는 `analyze.py`, 콘솔 기록은 `run.log`. 기존 데이터를 덮어쓰지 않는 새 빈 실험 폴더에서만 실행해야 한다.

## Oracle ROI 대 YOLO ROI

| 분류 | 수 | 프레임 |
|---|---:|---|
| 둘 다 성공 | {len(cat['both_success'])} | {', '.join(cat['both_success'])} |
| Oracle만 성공 | {len(cat['oracle_only'])} | {', '.join(cat['oracle_only'])} |
| YOLO만 성공 | {len(cat['yolo_only'])} | {', '.join(cat['yolo_only'])} |
| 둘 다 실패 | {len(cat['both_fail'])} | {', '.join(cat['both_fail'])} |

`frame_comparison.csv`는 모든 프레임의 oracle/YOLO PnP와 대칭 ADD, CAD 투영 bbox 대비 IoU, 중심 x·y 이동(px), 너비·높이 비율 및 confidence를 기록한다. CAD bbox는 amodal 투영이며 bbox GT도 pose 불확실성의 영향을 받는다. 기존에 66, 68, 70, 72, 73은 GT pose가 의심된다고 기록돼 있어 단일 원인을 단정하지 않는다.

PnP 성공 26장의 bbox IoU와 YOLO 명시적 대칭 ADD의 Pearson r={st['bbox_iou_vs_symmetry_ADD_pearson']:.3f}, Spearman ρ={st['bbox_iou_vs_symmetry_ADD_spearman']:.3f}. 극단 오차인 66을 뺀 25장에서는 Pearson r=-0.482, Spearman ρ=-0.582다(68은 PnP 실패로 원래 제외). 프레임들이 독립 표본이라는 보장이 없고 GT-derived bbox와 GT 오류의 공통 영향도 있어 인과관계나 통계적 일반화로 해석하지 않는다.

## ROI 배율 실험: 실사 개발27

| bbox 배율 | PnP | 명시적 대칭 ADD | ADD-S | 일반 ADD | XYZ 변화 중앙값 mm |
|---:|---:|---:|---:|---:|---:|
{roi}

PnP 실패는 모든 지표 분모 27에 실패로 포함한다. 1.20배에서 새로 성공: **{', '.join(good)}**, 새로 실패: **{', '.join(bad)}**. 이는 +3/-1의 순 2장 변화다. 1.00과 1.20의 PnP 실패는 모두 68이다. ADD-S 하락과 새로운 실패가 있어 이 설정을 확정 운영 정책으로 채택하지 않는다. `roi_comparison.csv`에는 모든 135회 프레임별 오차·mask 수·대칭 index·대칭 회전/이동 오차가 있다. `overlays/roi/`에는 135개 투영과 변한 4장의 `comparison_*.jpg`가 있다. 녹색은 GT pose 투영, 자홍은 예측 투영, 노랑은 1.5 pad를 적용한 ROI다.

XYZ 변화는 1.00배의 출력 그리드를 실제 이미지 좌표로 변환하고 다른 배율 XYZ를 **nearest-neighbor**로 재표본화해 두 예측 mask가 겹친 픽셀에서 front XYZ 간 유클리드 차이 중앙값을 구한 뒤 프레임별 중앙값의 중앙값을 표시했다. 1.00배와의 상대적 **예측 안정성**이며 GT XYZ 정확도나 실사 가림 픽셀의 타당성이 아니다. 공유 영역 선택과 양자화·재표본화에 영향을 받는다. 배율에 따라 공유 픽셀 수는 CSV에 기록했다.

## BB100 실패 45장: 프레임별 원인 조사

기존 공식 모델 oracle ROI의 BB100은 명시적 대칭 ADD 실패 45/100이고, 실패 45장 모두 PnP는 유효 pose를 반환했다. PnP 반환은 안정성이나 정확성의 증거가 아니므로 PnP 불안정 가능성을 배제하지 않는다. 기존 분석의 시간/근접 pose 연결 성분 31 그룹에서는 33/36장 실패, 221 그룹에서는 12/64장 실패했다. 이들은 동일 촬영 세션에 속하므로 독립 그룹 실험이 아니다. 반올림 1e-8 기준 동일 GT pose는 전체 36그룹, 실패 프레임이 포함된 그룹 19개이며 성공과 실패가 섞인 동일 pose 그룹은 3개다. 동일·연속 프레임의 높은 상관을 고려해야 한다.

실패 45장의 대칭 선택 index는 모두 1이고, 전체 성공 55장도 index 1이다. 실패의 대칭 회전 오차 중앙값 1.91°, 대칭 이동 오차 중앙값 25.61 mm, 성공의 중앙값은 각각 1.51°, 6.75 mm였다. 이는 기록된 GT와 공식 CAD 좌표계 기준 수치이며 GT pose의 실제 정확도 또는 camera calibration을 검증하지 않는다. 일반 ADD, ADD-S와 명시적 대칭 ADD는 서로 다른 지표이므로 직접 혼용하지 않는다.

45장의 기존 GT/예측 투영 crop을 5개 contact sheet로 모두 확인하고, 각 프레임에 mask·front/back XYZ 예측 패널을 붙인 `overlays/bb100_failures/000xxx.jpg`를 만들었다. 앞쪽 031–054는 표시된 crop에서 도구가 보이지 않는 실패가 이어지고, 067 이후에는 도구가 crop 안에 보이는 사례가 나타난다. **도구의 존재는 가림 여부·가림률이 아니며**, 겹친 선과 카메라 각도 때문에 가림 정도를 판정하지 않았다. 실패의 특정 원인을 실측할 근거는 부족하므로 각 프레임의 보정, GT pose, 회전, ROI, XYZ, PnP 원인은 `unknown`으로 기록했다. GT 투영 bbox를 이용한 oracle ROI 자체에도 GT 품질 한계가 있다. contact sheet는 위치 비교용이며 표면/픽셀 단위 정합 검수가 아니다.

`bb100_failure_analysis.csv`는 실패 45장과 성공 55장의 상태, 시간 그룹, 동일 pose 그룹, 오차, mask 픽셀 수, 대칭 index와 원인 미확정 상태를 기록한다. `failure_statistics.json`은 그룹별 실패 수, 상태별 수와 입력 SHA를 포함한다. 모델의 실제 visible GT mask가 없으므로 **가림 정도와 실패의 정량적 관계는 계산하지 않았다.**

## 판단과 후속 실험

- 1.20배는 개발셋에서 검토할 후보일 뿐이다. 독립 촬영 세션을 확보해 1.00/1.20을 사전 고정하고 동일 프레임·동일 평가기로 대칭 ADD, ADD-S, PnP와 악화 사례를 함께 비교한다. confidence·여러 배율을 같은 개발셋에 추가로 반복 조정하지 않는다.
- BB 그룹 31의 반복 실패는 보정 K,D, GT pose·시간 동기, 물체 형태와 oracle ROI를 별도 기준으로 우선 확인한다. 추론 overlay의 공통 경향만으로 보정 또는 GT 오류를 결론내리지 않는다.
- 독립 기준이 생기면 가림 조건별 정확한 visible 영역과 GT pose를 확인하고, 반복 PnP seed에 따른 분산과 HCCE XYZ를 분리 평가한다. 실사 영상에 CAD amodal을 visible GT로 적용하지 않는다.
- 배율 정책 채택이나 학습을 진행하기 전에 촬영 세션 단위 분리를 확보한다. 기존 데이터의 27장/100장 결과와 공식 BOP AR은 직접 비교하지 않는다.
''';(P/'연구_보고서.md').write_text(md,encoding='utf8');(P/'다음_실험.md').write_text('''# 다음 실험

1. 독립 촬영 세션에서 bbox 1.00/1.20 고정 후 paired 평가: 명시적 대칭 ADD, ADD-S, PnP, 프레임별 악화 사례를 모두 보고한다.
2. BB 그룹 31의 K,D, 시간 대응 및 GT pose를 외부 기준으로 검증한다. 재투영만으로 정확도를 확정하지 않는다.
3. 수동으로 검수한 소수의 visible mask 및 pose를 기준으로 도구 가림과 XYZ 오차를 별도 검증한다.
4. seed별 PnP 분산과 객체 좌표 변화·crop sensitivity를 분해해 원인을 확인한다.
''',encoding='utf8');print('gained',good,'lost',bad,'tool present',st['visual_tool_in_failure_crop'])
