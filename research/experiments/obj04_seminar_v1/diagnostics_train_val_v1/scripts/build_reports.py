"""검증된 실측 CSV에서 비교 그래프와 한국어 보고서 신규 생성."""
from pathlib import Path
import csv,json,datetime,hashlib,re,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];BASE=R.parent;TRAIN=BASE/'runs/baseline_seed42_20261103_214353/trained_only';PREV=BASE/'diagnostics_v1'
def rows(p):return list(csv.DictReader(p.open()))
def read(p):return json.loads(p.read_text())
def write(p,text):
 with p.open('x') as f:f.write(text)
def js(p,x):write(p,json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def csvout(p,data):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def link(p):return f'[{p.name}]({p})'
def num(x):return float(x)
X=rows(R/'train_val_xyz_comparison.csv');P=rows(R/'train_val_pose_metrics.csv');H=rows(R/'hcce_plane_error.csv');F=rows(R/'tables/frame_metadata.csv');E=rows(R/'tables/xyz_per_frame.csv');PP=rows(R/'tables/pose_per_frame.csv');BLOCK=rows(R/'tables/block_metrics.csv');REL=rows(R/'tables/relationships.csv');DONE=read(R/'checks/completion.json')
def xyz(s,b,scope='GTmask'):return next(x for x in X if x['split']==s and x['branch']==b and x['scope']==scope)
def plane(s,b,a,k):return next(x for x in H if x['split']==s and x['branch']==b and x['axis']==a and int(x['plane'])==k)
def pose(s,b):return next(x for x in P if x['split']==s and x['branch']==b)
TF=xyz('train','front');VF=xyz('val','front');TB=xyz('train','back');VB=xyz('val','back')
# Validate new Val metrics exactly against previous diagnostic values.
assert DONE['Val_front_ADD_max_difference_vs_previous']==0 and DONE['Val_XYZ_max_difference_vs_previous']==0
integrity={}
for name,path,root in [('originals',BASE/'checks/original_manifest.json',None),('seminar',BASE/'checks/artifact_manifest.json',BASE),('baseline',TRAIN.parent/'checks/final_artifact_manifest.json',TRAIN.parent),('diagnostics_v1',PREV/'checks/final_artifact_manifest.json',PREV)]:
 rec=read(path);bad=[]
 for x in rec:
  p=Path(x['path']) if root is None else root/x['path']
  if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=x['sha256']:bad.append(str(p))
 integrity[name]={'checked':len(rec),'mismatches':bad}
assert all(not x['mismatches'] for x in integrity.values()),integrity
integrity['final_gpu_compute_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
js(R/'checks/final_integrity.json',integrity)
# Descriptive GT marginal histogram comparison, not an independence test.
HIST=read(R/'checks/histograms.json');ds=[]
for branch in ('front','back'):
 for axis in 'xyz':
  a=np.array(HIST['counts'][f'train_{branch}_{axis}']['GT'],float);b=np.array(HIST['counts'][f'val_{branch}_{axis}']['GT'],float);a/=a.sum();b/=b.sum();m=(a+b)/2
  jsd=.5*(np.sum(a[a>0]*np.log(a[a>0]/m[a>0]))+np.sum(b[b>0]*np.log(b[b>0]/m[b>0])))
  ds.append({'branch':branch,'axis':axis,'bins':64,'total_variation':float(.5*np.abs(a-b).sum()),'JS_divergence_nats':float(jsd),'note':'pixel-pooled marginal, not image-conditional shift/independence test'})
csvout(R/'tables/GT_distribution_comparison.csv',ds)
# ADD failure groups. All source solver returns were valid; these are accuracy failures.
fg=[]
for s in ('train','val'):
 for b in ('front','back'):
  for passed in (True,False):
   pp=[p for p in PP if p['split']==s and p['branch']==b and (p['add_pass']=='True')==passed];look={int(e['frame']):e for e in E if e['split']==s and e['branch']==b and e['scope']=='GTmask'}
   fg.append({'split':s,'branch':b,'ADD_pass':passed,'frames':len(pp),'mean_xyz_error':float(np.mean([num(look[int(p['frame'])]['xyz_l2']) for p in pp])) if pp else None,'mean_inlier_ratio':float(np.mean([num(p['inlier_ratio_actual']) for p in pp])) if pp else None})
csvout(R/'tables/ADD_pass_failure_comparison.csv',fg)
# Graphs from actual data. No interpolation, smoothing or error bars implying independence.
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):
 try:return ImageFont.truetype(fontpath,n)
 except OSError:return ImageFont.load_default()
FT=font(24);FM=font(17);FS=font(14)
BLUE='#286da8';RED='#d45b43'
def canvas(title,subtitle,size=(1300,760)):
 im=Image.new('RGB',size,'white');d=ImageDraw.Draw(im);d.text((45,18),title,font=FT,fill='black');d.text((45,54),subtitle,font=FM,fill='#444444');return im,d
def axis(d,box,xmax,ymax,xticks=5,yticks=5):
 l,t,r,b=box
 for i in range(yticks+1):
  y=b-(b-t)*i/yticks;d.line((l,y,r,y),fill='#dddddd');d.text((l-70,y-10),f'{ymax*i/yticks:.2f}',font=FS,fill='black')
 for i in range(xticks+1):
  x=l+(r-l)*i/xticks;d.text((x-15,b+12),f'{xmax*i/xticks:.0f}',font=FS,fill='black')
 d.line((l,t,l,b,r,b),fill='black',width=2)
# 1. Four common-unit XYZ bars.
im,d=canvas('Train80 / Val20 XYZ error','epoch155 | batch4 AMP | GT mask & valid labels | native PBR units')
box=(120,125,1240,620);l,t,r,b=box;mx=max(num(VF['xyz_l2']),num(VB['xyz_l2']))*1.2
for i in range(6):
 y=b-(b-t)*i/5;d.line((l,y,r,y),fill='#dddddd');d.text((40,y-9),f'{mx*i/5:.1f}',font=FM,fill='black')
for i,(label,key,br) in enumerate([('Front L2','xyz_l2','front'),('Back L2','xyz_l2','back'),('Front Z MAE','z_MAE','front'),('Back Z MAE','z_MAE','back')]):
 x=l+60+i*275
 for j,(s,col) in enumerate([('train',BLUE),('val',RED)]):
  v=num(xyz(s,br)[key]);xx=x+j*90;y=b-v/mx*(b-t);d.rectangle((xx,y,xx+72,b),fill=col);d.text((xx,y-27),f'{v:.2f}',font=FM,fill=col)
 d.text((x-10,b+22),label,font=FM,fill='black')
d.text((150,690),'Train80 (fit diagnostic)',font=FM,fill=BLUE);d.text((600,690),'Val20 (development validation)',font=FM,fill=RED);im.save(R/'train_val_xyz_error.png')
# 2. Z plane 0..7 curves, separate branches.
im,d=canvas('Z HCCE plane error: Train vs Validation','Actual source FP16 sigmoid threshold; per-frame rates averaged',size=(1400,760))
for j,br in enumerate(('front','back')):
 l=100+j*680;t=150;r=l+570;b=590;d.text((l,102),br.upper(),font=FT,fill='black')
 for i in range(6):
  y=b-(b-t)*i/5;d.line((l,y,r,y),fill='#dddddd');d.text((l-60,y-10),f'{i*12}%',font=FS,fill='black')
 for s,col in [('train',BLUE),('val',RED)]:
  points=[]
  for k in range(8):
   v=num(plane(s,br,'z',k)['hard_bit_error'])*100;x=l+k/7*(r-l);y=b-v/60*(b-t);points.append((x,y));d.ellipse((x-4,y-4,x+4,y+4),fill=col);d.text((x-20,y-24 if s=='val' else y+10),f'{v:.1f}',font=FS,fill=col)
  d.line(points,fill=col,width=3)
 for k in range(8):d.text((l+k/7*(r-l)-5,b+35),str(k),font=FM,fill='black')
 d.text((l+180,b+70),'HCCE plane index',font=FM,fill='black')
d.text((180,705),'Train80',font=FM,fill=BLUE);d.text((800,705),'Val20',font=FM,fill=RED);im.save(R/'z_plane_error_comparison.png')
# 3. Mask vs XYZ scatter, same axes and independent units.
im,d=canvas('Mask IoU vs Front XYZ error','Each dot = one frame; Train fit differs from Val generalization');l,t,r,b=120,125,1230,620
for i in range(6):
 y=b-(b-t)*i/5;d.line((l,y,r,y),fill='#dddddd');d.text((45,y-10),str(i*12),font=FM,fill='black');x=l+(r-l)*i/5;d.text((x-20,b+20),f'{.85+.15*i/5:.2f}',font=FM,fill='black')
for e in E:
 if e['branch']=='front' and e['scope']=='GTmask':
  x=l+(num(e['mask_iou'])-.85)/.15*(r-l);y=b-num(e['xyz_l2'])/60*(b-t);c=BLUE if e['split']=='train' else RED;d.ellipse((x-4,y-4,x+4,y+4),fill=c)
d.text((520,680),'Mask IoU',font=FM,fill='black');d.text((125,90),'XYZ mean L2 (PBR native units)',font=FM,fill='black');d.text((150,720),'Train80',font=FM,fill=BLUE);d.text((850,720),'Val20',font=FM,fill=RED);im.save(R/'mask_vs_xyz_comparison.png')
# 4. Normalized GT Z marginal histograms, same 64 bins for both splits.
im,d=canvas('GT Z distribution (normalized histogram)','64 fixed bins, pixel-pooled probabilities; marginal overlap does not rule out conditional shift',size=(1400,760))
for j,br in enumerate(('front','back')):
 l=100+j*680;t=145;r=l+570;b=590;d.text((l,100),br.upper(),font=FT,fill='black');edges=np.array(HIST['edges']['z']);curves={s:np.array(HIST['counts'][f'{s}_{br}_z']['GT'],float) for s in ('train','val')};curves={s:q/q.sum() for s,q in curves.items()};mx=max(max(q) for q in curves.values())*1.15
 for i in range(6):
  y=b-i/5*(b-t);d.line((l,y,r,y),fill='#dddddd');d.text((l-65,y-8),f'{mx*i/5:.2f}',font=FS,fill='black')
 for s,c in [('train',BLUE),('val',RED)]:
  q=curves[s];pts=[(l+(k+.5)/64*(r-l),b-v/mx*(b-t)) for k,v in enumerate(q)];d.line(pts,fill=c,width=3)
 for k in range(5):d.text((l+(r-l)*k/4-20,b+20),f'{edges[0]+(edges[-1]-edges[0])*k/4:.1f}',font=FS,fill='black')
 d.text((l+190,b+60),'GT object Z',font=FM,fill='black')
d.text((180,705),'Train80',font=FM,fill=BLUE);d.text((800,705),'Val20',font=FM,fill=RED);im.save(R/'gt_z_histogram_comparison.png')
# 5. Front coordinate error by generation block.
im,d=canvas('Front XYZ error by generation block','10 frames/block; Train blocks blue / Val blocks red');l,t,r,b=100,130,1250,610;mx=40
for i in range(5):
 y=b-i/4*(b-t);d.line((l,y,r,y),fill='#dddddd');d.text((40,y-8),str(i*10),font=FM,fill='black')
for v in BLOCK:
 if v['branch']=='front':
  k=int(v['block']);x=l+25+k*112;y=b-num(v['xyz_l2'])/mx*(b-t);c=BLUE if v['split']=='train' else RED;d.rectangle((x,y,x+70,b),fill=c);d.text((x,y-28),f'{num(v["xyz_l2"]):.2f}',font=FM,fill=c);d.text((x+25,b+20),str(k),font=FM,fill='black')
d.text((470,680),'Generation block ID',font=FM,fill='black');im.save(R/'block_xyz_error.png')
# Precision control and diagnosis records.
PC=rows(R/'tables/precision_control.csv');bitdiff=sum(int(p['sign_vs_actual_bit_differences_GTmask']) for p in PC);bitcount=sum(int(p['all_bit_count_GTmask']) for p in PC)
ratioF=num(VF['xyz_l2'])/num(TF['xyz_l2']);ratioB=num(VB['xyz_l2'])/num(TB['xyz_l2']);ratioZF=num(VF['z_MAE'])/num(TF['z_MAE']);ratioZB=num(VB['z_MAE'])/num(TB['z_MAE'])
hypotheses=[{'hypothesis':'A Train/Val 모두 큰 XYZ 학습 실패','assessment':'주 설명으로는 지지 약함, 세밀한 plane 학습 잔차는 확인','evidence':f'Train Front L2={num(TF["xyz_l2"]):.4f}, Z MAE={num(TF["z_MAE"]):.4f}; Val은 {ratioF:.2f}/{ratioZF:.2f}배','limitation':'Train Front ADD도59/80으로 완전 적합은 아니며 fine plane 오류 남음'}, {'hypothesis':'B 과적합·블록 일반화 실패','assessment':'현재 네 가설 중 가장 강한 지지','evidence':'Train Front ADD59/80 대 Val1/20, 두 Val 블록 모두 모든 Train블록보다 Front XYZ오차 큼','limitation':'단일 seed/checkpoint·Val2블록. 원인이 초기화/증강/표본량 중 무엇인지는 미확정'}, {'hypothesis':'C Mask·XYZ 학습 불균형','assessment':'현상은 지지, 최적화 불균형 원인은 미확정','evidence':f'Mask IoU {num(TF["mask_iou"]):.4f}/{num(VF["mask_iou"]):.4f}지만 XYZ gap {ratioF:.2f}배','limitation':'loss branch gradient/scale 실험 없음. Train coarse 좌표는 학습됨'}, {'hypothesis':'D Z coarse plane 민감도','assessment':'Val에서 강하게 지지; Train coarse는 상대적으로 양호','evidence':f'Front Z0 hard error {num(plane("train","front","z",0)["hard_bit_error"])*100:.2f}%/{num(plane("val","front","z",0)["hard_bit_error"])*100:.2f}%; GT plane0치환 Val Z MAE 약3.22','limitation':'Oracle 치환 진단이며 실제 모델 개선 아님; fine plane도 미학습 잔차'}]
csvout(R/'tables/hypothesis_assessment.csv',hypotheses)
# Report table builders.
xt='\n'.join(f"| {s} | {br} | {num(xyz(s,br)['mask_iou']):.6f} | {num(xyz(s,br)['xyz_l2']):.4f} | {num(xyz(s,br)['x_MAE']):.4f} | {num(xyz(s,br)['y_MAE']):.4f} | {num(xyz(s,br)['z_MAE']):.4f} | {num(xyz(s,br)['raw_target_MAE']):.6f} | {num(xyz(s,br)['hard_bit_error'])*100:.2f}% |" for s in ('train','val') for br in ('front','back'))
pt='\n'.join(f"| {v['split']} | {v['branch']} | {v['PnP_valid']}/{v['frames']} | {v['ADD_pass']}/{v['frames']} ({num(v['ADD_rate'])*100:.2f}%) | {num(v['add_mean']):.4f}/{num(v['add_median']):.4f} | {num(v['rotation_deg_mean']):.4f} | {num(v['translation_mean']):.4f} | {num(v['inlier_ratio'])*100:.2f}% |" for v in P)
zt=[]
for br in ('front','back'):
 for k in range(8):
  a=plane('train',br,'z',k);b=plane('val',br,'z',k)
  zt.append(f"| {br} | {k} | {a['decoder_binary_weight']} / {a['maximum_cascade_code_change']} | {num(a['hard_bit_error'])*100:.2f}% | {num(b['hard_bit_error'])*100:.2f}% | {num(a['GT_positive_fraction'])*100:.1f}/{num(b['GT_positive_fraction'])*100:.1f}% | {num(a['pred_positive_fraction'])*100:.1f}/{num(b['pred_positive_fraction'])*100:.1f}% | {num(b['Oracle_MAE_reduction']):.4f} |")
ct='\n'.join(f"| {s} | {br} | {num(plane(s,br,'z',0)['abs_raw_margin_mean']):.4f} | {num(plane(s,br,'z',0)['sigmoid_mean']):.4f} | {num(plane(s,br,'z',0)['confidence_mean']):.4f} |" for s in ('train','val') for br in ('front','back'))
xytable='\n'.join(f"| {br} | {a.upper()} | {num(plane('train',br,a,0)['hard_bit_error'])*100:.2f}% | {num(plane('val',br,a,0)['hard_bit_error'])*100:.2f}% |" for br in ('front','back') for a in 'xyz')
blocktable='\n'.join(f"| {v['block']} | {v['split']} | {num(v['xyz_l2']):.4f} | {num(v['z_MAE']):.4f} | {num(v['mask_iou']):.4f} | {v['ADD_pass']}/{v['frames']} |" for v in BLOCK if v['branch']=='front')
ht='\n'.join('| '+' | '.join(v.values())+' |' for v in hypotheses)
now=datetime.datetime.now(datetime.timezone.utc);elapsed=(now-datetime.datetime.fromtimestamp(1793712981,datetime.timezone.utc)).total_seconds()/60
frontz=plane('val','front','z',0);backz=plane('val','back','z',0)
text=f'''# Train80 / Val20 XYZ 학습 실패와 일반화 실패 원인 분석

작성: {now.astimezone(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}

**결론: 현재 checkpoint는 Train 좌표를 상당히 학습했지만, 분리된 Val 블록에서 좌표 오차가 급증한다. 전반적 XYZ 학습 실패보다 과적합·합성 블록 일반화 실패의 증거가 강하며, Val Z plane0의 coarse 오류가 특히 크다.** 다만 작은 Train 잔차와 fine-plane 학습 문제도 남아 있다.

- 새 결과: `{R}`
- 모델: `{TRAIN/'checkpoints/best_val_loss.pt'}`, epoch155, 재학습 없음
- 주 조건: Train80/Val20 모두 batch4, CUDA AMP float16, model.eval/inference_mode
- GT BBox oracle crop·BGR 정규화·기존 XYZ 라벨·원본 HCCE 복호화 동일
- 계산2회 합계 약{read(R/'sign_threshold_control/checks/completion.json')['elapsed_seconds']+DONE['elapsed_seconds']:.2f}초(첫 sign-threshold 대조 포함); 보고서 생성까지 전체 작업 약{elapsed:.2f}분 /45분. 계산 wall time은 로딩·CPU 집계를 포함하며 GPU kernel 시간은 아니다.
- 실제 SMT27·YOLO·학습 optimizer는 사용하지 않았다. 기존 전용 venv 재사용, 설치·다운로드 없음.

## 1. 요청 질문에 대한 답

1. **어느 쪽 XYZ 오류가 큰가:** Val이 크다. Front L2 Train4.2057→Val29.4046({ratioF:.2f}배), Back4.5603→29.1845({ratioB:.2f}배)다.
2. **Train에서도 Z 오류가 심각한가:** Train Front/Back Z MAE1.3698/1.5070으로 Val16.3539/15.7365보다 훨씬 작다. coarse Z0 오류도 Train2.75%/3.57% 대 Val44.57%/43.12%다. Train fine Z planes5–7에는 약40–50% 오류가 남지만 영향 크기가 다르다. Train이 완벽하다는 뜻은 아니다.
3. **큰 좌표 오차에 기여하는 plane:** Val Z plane0이다. 이 한 plane만 GT로 치환한 Oracle에서 Front Z MAE reference {num(frontz['axis_reference_MAE']):.4f}→{num(frontz['axis_MAE_after_GT_plane_Oracle']):.4f}, Back {num(backz['axis_reference_MAE']):.4f}→{num(backz['axis_MAE_after_GT_plane_Oracle']):.4f}다. 실제 모델 출력의 개선 수치가 아니다.
4. **학습 부족과 과적합 중 증거가 강한 쪽:** 현재 조건에서는 과적합·블록 일반화 격차 쪽이다. Train Front ADD59/80 대 Val1/20, Train Z coarse 오류가 낮다. fine-plane 잔차가 있어 학습 부족 가능성 전체를 배제하지는 않는다. 한 checkpoint 비교로 epochs·Loss·초기화 중 인과 원인을 확정할 수 없다.
5. **다음 실험에서 딱 하나 바꿀 것:** **학습 crop의 동기화된 in-plane 회전 증강만 추가한다(고정 정책: uniform[-20°,20°]).** RGB·Mask·Front/Back XYZ·유효영역에 같은 affine를 적용하고 label은 nearest, HCCE는 변환된 XYZ에서 재생성한다. 나머지 초기화·Loss·분할·seed·step budget·평가·선택 규칙은 유지한다. 미실행 가설 검증이며, 3D viewpoint 일반화를 보장하지 않는다.
6. **내일 확정적으로 말할 결론:** “좌표는 Train에서는 상당히 맞지만 Val 블록에서 크게 틀리며, 특히 Z coarse plane 일반화가 나쁘다. Mask 양호만으로 Pose 일반화를 판단할 수 없다.” 독립 테스트 정확도·실사 정확도·개선 모델 확보는 주장하지 않는다.

## 2. 동일 조건 및 측정 정의

| 항목 | 실제 설정 |
|---|---|
| checkpoint | 기존 random-initialized ResNet34 학습의 epoch155, strict state loading |
| Train / Val | 80/20, Train블록0,1,2,3,5,6,7,8; Val블록4,9. 프레임·블록 교집합 없음 |
| crop | 원본 `pad_square_fp32` pad1.5, `crop_square_resize`, RGB256, Mask/XYZ128 |
| RGB | BGR, mean(0.406,0.456,0.485), std(0.225,0.224,0.229), 학습과 동일 |
| 좌표 | object model XYZ, min/size 동일. PBR 원본 단위, 절대 물리치수 독립 계측 없음 |
| 주 오차 영역 | `GTmask & Front/Back nonzero-label validity`, 양 branch 동일 |
| 보조 영역 | PredMask∩유효label, GT/Pred 교집합, GTmask3×3 erode; 동일 CSV에 보존 |
| Mask IoU | 전체 crop의 GTmask 대 predmask. 좌표 유효영역으로 IoU를 유리하게 제한하지 않음 |
| XYZ | 실제 원본 AMP decoder 출력 대 동일픽셀 GT 양자화 XYZ |
| raw target | 네트워크 raw code 출력 대 `2*hcce_encode-1` MAE, 총 Loss 아님 |
| hard bit | 원본과 동일 FP16 sigmoid>0.5 대 GT encode>0.5 |
| Pose | 실제 원본 RANSAC EPNP 반환,150iterations,1.5 원본px, seed42+frame, branch별 PnP |
| ADD | CAD1439점, threshold13.9378694855, 모든프레임 분모, GT행렬 복사 안 함 |
| 통계 | 주 표는 프레임별 평균의 split 평균. pixel-weighted 열도 별도 보존 |

GT BBox ROI는 모든 조건의 공통 Oracle 입력이다. 예측XYZ·예측Mask로 구한 Pose는 **oracle ROI 조건의 모델 결과**이며 E2E 검출 성능이 아니다. GT bit 치환은 추가로 GT를 쓴 **Oracle 민감도 진단**이다. Train 결과는 학습 적합도이며 일반화 성능으로 보고하지 않는다.

근거: {link(R/'checks/config.json')}, {link(R/'checks/split.json')}, `{R/'scripts/measure_source_threshold.py'}`, 이전 학습 `scripts/support.py`, 원본 `/RESEARCH_DATA/HCCEPose/HccePose/network_model.py`, `bop_loader.py`, `PnP_solver.py`.

## 3. Train / Val XYZ·Mask·raw 비교

| split | branch | Mask IoU | XYZ L2 | X MAE | Y MAE | Z MAE | raw-target MAE | hard bit 오류 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{xt}

GTmask 유효픽셀은 Train261,419, Val64,635이다. Front Z MAE는Val이Train의{ratioZF:.2f}배, Back은{ratioZB:.2f}배다. Mask IoU 차이에 비해 좌표 격차가 크다. raw MAE는 회귀 출력의 오차이고, hard bit 오류는24개 plane의 동일 가중치 평균이다. 저가중 fine plane 오류가 커도 전체 XYZ에 미치는 영향은 coarse 오류보다 작을 수 있다.

보조 Mask 영역에서도 같은 비교를 모두 저장했다. GT 없는 background 좌표를 임의로 정답화하지 않고 유효label과 교집합했다. GTmask의 명시적 무효 label은 좌표오차 계산에서 제외하고 프레임별 개수를 기록했다.

그림: {link(R/'train_val_xyz_error.png')}, {link(R/'mask_vs_xyz_comparison.png')}.

## 4. 실제 Train / Val Pose

| split | branch | 유효 PnP | ADD 통과 | ADD 평균/중앙 | RE 평균(deg) | TE 평균 | 실제 inlier 비율 |
|---|---|---:|---:|---:|---:|---:|---:|
{pt}

100장×2branch 모두 실제 PnP를 실행했다. 유효 반환은 모두100%였지만 정확도는 크게 달랐다. Front Train ADD59/80=73.75%, Val1/20=5%다. Back Train65/80=81.25%, Val1/20=5%다. Back 모델을 GT로 프레임별 선택하거나 더 높은 Train 수치를 일반화 정확도로 보고하지 않는다.

원본 Back PnP의 reported inlier는 Front로 재집계되는 기존 문제를 유지하고 로그에 남겼다. 표는 원본solver에 실제 입력된 branch 3D점 기준으로 독립 재투영한 inlier 비율이다. 원본 파일은 수정하지 않았다. RANSAC raw inlier와 반환 후 재집계 inlier도 다를 수 있다.

Val20의 Front ADD·XYZ는 이전 진단과 프레임별 최대차이0으로 재현됐다. 이번 Train Mask 수치는 저장 checkpoint의 새 추론에서 직접 측정한 값이다.

## 5. HCCE plane 순서·가중치·민감도

원본 `hcce_decode`의 축별 순서는0→7이다. HCCE hard plane을g, 복원 binary를b라 하면:

```text
b0 = g0
bk = gk XOR b(k-1)
q = 128*b0 + 64*b1 + 32*b2 + 16*b3 + 8*b4 + 4*b5 + 2*b6 + b7
XYZ_axis = q/255 * size_axis + min_axis
```

따라서 gk 하나의 flip은 binary k..7에 전파될 수 있다. 복호화 자리 가중치128..1을 그 HCCE plane의 독립적인 오차 가중치로 간주하면 안 된다. 이론상 최대 code 변화는 `2^(8-k)-1`이다. Z에서 plane0 최대65.3153단위 대 plane7 최대0.2561단위이며, 실제 영향은 표본에 따라 달라진다.

### Z plane0~7 전체 비교

| branch | plane | binary 자리 / 최대cascade(code) | Train오류 | Val오류 | GT=1 비율 T/V | Pred=1 비율 T/V | Val GT-plane치환 MAE 감소 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(zt)}

감소량은 원본과 정수 복호화가 일치하는 FP32 reference에서 계산했다. 한 plane의 GT hard bit만 치환하고 나머지는 예측을 유지했다. 이 감소량을 실제 모델 성능이나 그 plane의 독립적 인과 기여율로 해석하지 않는다. plane별 감소량은 상호작용 때문에 단순 합산하지 않는다.

Front Val Z plane0 치환 reference MAE 감소는{num(frontz['Oracle_MAE_reduction']):.4f}({num(frontz['Oracle_MAE_reduction'])/num(frontz['axis_reference_MAE'])*100:.1f}% 감소)이나 plane7은{num(plane('val','front','z',7)['Oracle_MAE_reduction']):.4f}뿐이다. **오류율이 약50%인 fine plane과 약45%인 coarse plane을 동일한 Pose 병목으로 취급하면 안 된다.**

Train의 coarse Z0는 낮은 오류지만 Z5–7은 여전히 약40–50%이다. 이 fine-scale 학습 잔차는 존재하며, Train이 완전히 학습됐다는 주장을 제한한다. Val의 큰 Z 오차를 가장 강하게 설명하는 민감도는 plane0에서 보인다.

그림: {link(R/'z_plane_error_comparison.png')}.

### 같은 coarse plane의 X/Y 비교

| branch | axis plane0 | Train 오류 | Val 오류 |
|---|---|---:|---:|
{xytable}

Val의 coarse 오류는 Z에서 특히 크며 X/Y도Train보다 증가한다. Z만의 구현 오류가 확정됐다는 뜻은 아니다.

### GT/Pred 비율·margin·한쪽 출력

| split | branch Z0 | mean abs raw margin | mean sigmoid | heuristic confidence |
|---|---|---:|---:|---:|
{ct}

Z0의 GT/Pred=1 비율은대체로50% 부근인데 Val error는약44%이다. 즉 coarse plane의 **전역적인 한쪽 출력 붕괴보다 이미지별/픽셀별 대응의 일반화 실패**가 관측된다. sigmoid confidence는L1 code 회귀 출력에서 계산한 heuristic일 뿐 보정된 확률이 아니다. margin이 남아 있어도 정확한 bit라는 보장은 없다.

반면 Front Z7은 Val Pred=1이{num(plane('val','front','z',7)['pred_positive_fraction'])*100:.3f}%로 한쪽 출력이 강하다. Train은{num(plane('train','front','z',7)['pred_positive_fraction'])*100:.3f}%이다. 단일방향 flag는사전 정의한 pooled Pred≤1% 또는≥99% 기준이고, GT 비율도 함께 확인했다. 이 fine-plane 현상을 coarse Z0의 실패와 혼동하지 않는다. 모든축·plane의 GT/Pred비율·raw mean/std·margin·confidence·TP/TN·balanced accuracy·프레임별 한쪽 출력 수는CSV에 있다.

## 6. FP16 임계값 정밀도 대조

첫 측정은기존 진단의 `raw>0` hard-bit 표현을 사용했다. 정밀도 점검에서FP16 sigmoid가0.5로 round되는 작은 양수는 원본의 `sigmoid(raw)>0.5`와 다를 수 있음을 확인했다. 원본 dtype/threshold로 재측정한 결과를 **현재 루트의 주 CSV**로 게시했다.

GTmask 유효영역의 총{bitcount:,}bit 비교 중 {bitdiff:,}개({bitdiff/bitcount*100:.4f}%)가 sign 대 실제 threshold에서 달랐다. 첫 결과는 `sign_threshold_control/`에 보존했다. 이 차이 때문에 앞선 진단의 Z0 약44.55%가이번 실제threshold에서44.57%로 정밀 보정되지만, 실제 AMP XYZ·Pose는 처음부터 원본 출력이므로 이전 진단 대비 최대차이0이다.

수정된 실제 hard bit와 원본 torch decoder의 정수 출력은 최대차이0이다. 이 reference의 XYZ와 원본 AMP XYZ 원소 최대차이는{DONE['max_AMP_vs_FP32_xyz_element_difference']:.6f}로, Val의29단위 오차를 설명하는 크기가 아니다. 첫 sign-threshold 대조의 큰 최대차이를 순수 AMP 좌표 연산 오차로 잘못 해석하지 않는다.

이 정밀도 대조는 새 증거를 보존하기 위한 것이며 소스·학습·checkpoint 수정이 아니다. 근거: {link(R/'tables/precision_control.csv')}, {link(R/'checks/publishing_policy.json')}.

## 7. 가설 판정

| 가설 | 판정 | 실제 증거 | 한계 |
|---|---|---|---|
{ht}

**현재 가장 강한 판정은 B와 D의 조합**이다. C는 출력 품질 격차의 현상으로 관측됐지만 gradient 경쟁이나 Loss 불균형이 원인인지 측정하지 않았다. A의전체적 학습 실패 설명은약하나 fine plane 잔차와 일부Train Pose 실패는 남는다.

## 8. 가벼운 추가 진단

### 생성 블록별 Front 결과

| block | split | XYZ L2 | Z MAE | Mask IoU | ADD 통과 |
|---|---|---:|---:|---:|---:|
{blocktable}

Train 블록 Front 평균오차 범위3.1682~6.8858인 반면 Val블록4는27.7782, 블록9는31.0311이다. 한두 Val 프레임만의 문제로 설명되지 않는다. 블록4 Mask IoU0.9613도 XYZ 일반화를 보장하지 못했다. 같은블록의10프레임은 상관된 표본이므로100개 독립 장면으로 해석하지 않는다.

GT XYZ분포는같은64개 model-axis bin을사용해분할별 합이1이되게 정규화했다. Z0 GT positive가 Train/Val 모두대체로50%여서 단순한클래스불균형으로 coarse gap을 설명하기 어렵다. **marginal 분포가 비슷하더라도 RGB-좌표의 조건부 분포/시점 차이는 배제할 수 없다.** XYZ축별분포·TV·JS 요약은CSV에 보존했고 정식 두분포 검정/인과 판정으로 사용하지 않았다.

GT object-Z와 camera-Z각도 범위는 Train58.35~125.49deg, Val60.32~121.18deg로 겹친다. 이것만으로 전체6D포즈 분포가같다고 할 수 없다. Front XYZ오차와 이각도의Pearson r은 Train-0.0391, Val0.0833이다. 가시율은모두약0.997~1.0으로 변동이작아 가림 효과를검증하기에부족하다.

Front XYZ오차와PnP실제inlier비율 r은Train-0.4362, Val-0.3354이며, ADD와는Train0.3292, Val0.7858이다. 같은 생성블록에상관된 기술통계다. 유효PnP반환 실패는0이었으므로 '실패한PnP'는여기서ADD 기준 미달로구분한다. ADD통과/실패별 좌표·inlier 집계도추가했다.

그림: {link(R/'gt_z_histogram_comparison.png')}, {link(R/'block_xyz_error.png')}. 근거: `tables/GT_distribution_comparison.csv`, `tables/GT_distribution_per_frame.csv`, `tables/relationships.csv`, `tables/ADD_pass_failure_comparison.csv`.

## 9. 다음 실험에서 한 가지 변경

**선택: 학습 crop에 동기화된 in-plane 회전 증강만 추가한다.** 이는현재 Baseline에 증강이없고 블록 일반화격차가 크다는 관찰에근거한 **다음 대조 실험 제안**이다. 개선효과는 미확인이다.

- 고정정책: 각Train crop의회전각 uniform[-20°,20°], seed42. RGB bilinear, Mask/XYZ/support nearest. HCCE는변환후XYZ에서생성한다. 모델좌표값자체는회전하지않고픽셀위치만동기변환한다.
- 고정할것: Train80/Val20및블록, random init ResNet34, batch4/crop256, Adam lr0.0002, 최대195epoch/3900step, Loss수식/타입호환, checkpoint후보epoch≥1/ValLoss규칙, Val無증강/동일PnP조건.
- 평가: 기존Front 주지표와 Z0 bit/XYZ를함께보고한다. 추가한증강에의한학습데이터적합도변화도Train에서측정한다. 새독립test성능이라고부르지않는다.
- 사전확인: CPU에서RGB·Mask·XYZ의같은affine와유효support정합성을검사한다. crop K가필요한augmented PnP를계산한다면K도같은affine로변환한다. 원본GT는수정하지않는다.
- 이번에는pretrained·Loss가중치·epochs·PnPthreshold를동시에바꾸지않는다. 회전증강은3Dout-of-plane시점변화를완전히만들지못하므로실패하면그대로보고한다.

이제안은이번에실행하지않았다. 기존PnPBack/Mixed inlier집계문제의정식수정도별도회귀작업으로분리한다. Front-only주평가와무관한수정을동시에섞지않는다.

## 10. 산출물·보존·재현

| 파일 | 내용 |
|---|---|
| {link(R/'train_val_xyz_comparison.csv')} | Train/Val·Front/Back·Mask영역별집계, 주표프레임평균과pixel가중치구분 |
| {link(R/'train_val_pose_metrics.csv')} | 실제200회PnP의split·branch집계 |
| {link(R/'hcce_plane_error.csv')} | 96개split/branch/axis/plane집계 |
| {link(R/'tables/xyz_per_frame.csv')} | 프레임별좌표·raw·bit오차 |
| {link(R/'tables/plane_per_frame.csv')} | 프레임별plane비율·margin·Oracle민감도 |
| {link(R/'tables/pose_per_frame.csv')} | 실제PnP오차·실패원인·inlier수 |
| {link(R/'checks/final_integrity.json')} | 이전자산SHA-256보존확인 |

보존대조: 원본740파일, 이전세미나366파일, Baseline61파일, 기존diagnostics_v1의102파일 모두해시일치. 기존checkpoint hash도실행전후동일하다. GPU종료query: `{integrity['final_gpu_compute_processes'] or 'compute process 없음'}`. 공유GPU다른compute PID가보이면계산을중단하는guard를배치전/프레임중간에적용했다.

재사용소스는이전 `diagnostics_v1/scripts/diagnose.py`와학습 `scripts/support.py`이다. 이번 실행코드는 `{R/'scripts/measure_source_threshold.py'}`이며 `logs` 원문을보존했다. 스크립트에는이실험의절대경로가들어있어재현시새root로경로설정을갱신해야한다. 기존출력에그대로재실행하지않는다. CPUreport/graph생성은 `{R/'scripts/build_reports.py'}`이다.

미수행: 재학습, YOLO, 실사27평가/튜닝, 새로운seed, loss-gradient실험, 독립test, 실제회전증강학습. 따라서새모델성능향상은주장하지않는다.
'''
# Improve Korean word spacing in repeated technical strings without changing code names or numbers.
for a,b in [('Val無증강','Val 무증강'),('plane0의','plane0의')]:text=text.replace(a,b)
write(R/'reports/Train_Val_XYZ_원인분석.md',text)
summary=f'''# 세미나 추가 진단 요약 — Train 적합도 대 Val 일반화

**핵심 결론:** 좌표 학습이 전반적으로 실패한 상태보다 **분리된 합성 블록에서 좌표 일반화가 실패한 상태**라는 증거가 강하다. Z coarse plane0의 격차가 특히 크다. 재학습은 하지 않았다.

| 지표 | Train80 — 학습 적합도 | Val20 — 개발 검증 |
|---|---:|---:|
| Mask IoU | {num(TF['mask_iou']):.6f} | {num(VF['mask_iou']):.6f} |
| Front XYZ L2 | {num(TF['xyz_l2']):.4f} | {num(VF['xyz_l2']):.4f} |
| Back XYZ L2 | {num(TB['xyz_l2']):.4f} | {num(VB['xyz_l2']):.4f} |
| Front Z MAE | {num(TF['z_MAE']):.4f} | {num(VF['z_MAE']):.4f} |
| Front Z0 bit 오류 | {num(plane('train','front','z',0)['hard_bit_error'])*100:.2f}% | {num(frontz['hard_bit_error'])*100:.2f}% |
| Front ADD@0.1d | 59/80 (73.75%) | 1/20 (5%) |
| Front 실제inlier 비율 | 13.42% | 4.01% |

같은epoch155·batch4·AMP·GTBBox crop으로비교했다. Train수치는일반화성능이아니다. Val의기존XYZ/Pose수치는프레임별최대차이0으로재현됐다.

## 슬라이드 1 — Train에서는 맞고 Val에서 좌표가 틀린다

- 핵심문장: “Front XYZ 오차가 Train4.21에서 Val29.40으로약{ratioF:.2f}배증가했다.”
- 숫자: Mask0.9740/0.9358, FrontADD73.75%/5%.
- 그림: {link(R/'train_val_xyz_error.png')}, {link(R/'mask_vs_xyz_comparison.png')}.
- 설명: 학습적합도와개발검증을구분한다. Train도완벽하지않다. 일반화격차의인과원인은아직확정하지않았다.

## 슬라이드 2 — Z coarse plane의 일반화가 특히 나쁘다

- 핵심문장: “Z0 오류는 Train2.75%에서 Val44.57%로급증한다.”
- 숫자: BackZ0도3.57%/43.12%. ValFrontZ0만GTbit치환하면referenceZMAE16.3542→3.2228.
- 그림: {link(R/'z_plane_error_comparison.png')}.
- 설명: bit0은뒤binary자릿값에도오류가전파된다. fineplane의50%오류와같은영향이아니다. GT치환은Oracle이며개선모델결과가아니다.

## 슬라이드 3 — 다음에는 회전 증강 하나만 바꾼다

- 핵심문장: “다음 대조는다른설정을고정하고Train crop의동기화된±20°회전증강만추가한다.”
- 숫자: Train8블록의Front평균오차3.17~6.89, Val2블록27.78/31.03.
- 그림: {link(R/'block_xyz_error.png')}, {link(R/'gt_z_histogram_comparison.png')}.
- 설명: 시점변화에대한대조가설이며효과는미확인이다. GTlabel/Mask에같은affine를적용하고HCCE를재생성한다. 실사27은튜닝에사용하지않는다.

## 발표 시 제한

- “전체XYZ학습완료”가아니다. TrainADD59/80, fineZplane의학습잔차가남는다.
- “증강으로정확도가올랐다”가아니다. 제안만했고추가학습은실행하지않았다.
- “일반화실패의원인은Loss/초기화다”라고확정하지않는다. 단일checkpoint·Val2블록의증거다.
- FP16실제sigmoid threshold와raw>0의작은차이를분리해실제threshold결과를주본으로게시했다. 기존Pose수치는변하지않는다.

상세보고서: {link(R/'reports/Train_Val_XYZ_원인분석.md')}
'''
write(R/'reports/세미나_추가진단_요약.md',summary)
for p in (R/'reports').glob('*.md'):
 links=re.findall(r'\]\((/[^)]+)\)',p.read_text());assert all(Path(x).is_file() for x in links),(p,[x for x in links if not Path(x).is_file()])
js(R/'checks/report_validation.json',{'report_files':2,'all_markdown_links_resolve':True,'task_elapsed_minutes_at_report':elapsed,'actual_threshold':'FP16 sigmoid>0.5','prior_xyz_and_pose_reproduced':True,'source_no_training':True})
print(json.dumps({'reports':[str(p) for p in (R/'reports').glob('*.md')],'elapsed_minutes':elapsed,'integrity':integrity,'GT_distributions':ds},ensure_ascii=False,indent=2))
