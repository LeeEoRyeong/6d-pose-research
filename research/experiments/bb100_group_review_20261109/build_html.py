import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;K=P/'package'
def read(p):
 with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
d={'groups':read(K/'group_metadata.csv'),'frames':read(K/'frame_review_template.csv'),'metrics':read(K/'motion_diagnostics.csv')};assert len(d['groups'])==36 and len(d['frames'])==len(d['metrics'])==100
text=(P/'index_template.html').read_text(encoding='utf8');assert text.count('__DATA__')==1;(K/'index.html').write_text(text.replace('__DATA__',json.dumps(d,ensure_ascii=False).replace('<','\\u003c')),encoding='utf8');print('HTML READY')
