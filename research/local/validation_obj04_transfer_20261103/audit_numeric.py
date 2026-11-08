"""원본을 읽기만 하는 표준 라이브러리 기반 obj04 수치 감사."""
import ast
import hashlib
import json
import math
from pathlib import Path
import struct

SOURCE = Path('/RESEARCH_DATA/HCCEPose_obj04_transfer')
OUT = Path(__file__).resolve().parent


def npy_matrix(path):
    with path.open('rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('NPY magic 오류')
        major, minor = f.read(2)
        length = struct.unpack('<H' if major == 1 else '<I', f.read(2 if major == 1 else 4))[0]
        h = ast.literal_eval(f.read(length).decode('latin1' if major < 3 else 'utf8').strip())
        if h['shape'] != (4, 4) or h['descr'] not in ('<f8', '<f4', '>f8', '>f4'):
            raise ValueError(f'지원하지 않는 안전 NPY 형식: {h}')
        values = struct.unpack(h['descr'][0] + ('16d' if h['descr'][2:] == '8' else '16f'), f.read())
        matrix = [[values[j*4+i] if h['fortran_order'] else values[i*4+j]
                   for j in range(4)] for i in range(4)]
    return matrix, h


def jpeg_size(path):
    with path.open('rb') as f:
        if f.read(2) != b'\xff\xd8':
            raise ValueError('JPEG 아님')
        while True:
            b = f.read(1)
            if not b:
                raise ValueError('SOF 없음')
            if b != b'\xff':
                continue
            marker = f.read(1)
            while marker == b'\xff':
                marker = f.read(1)
            m = marker[0]
            if m in (0xd8, 0xd9, 0x01) or 0xd0 <= m <= 0xd7:
                continue
            n = struct.unpack('>H', f.read(2))[0]
            data = f.read(n-2)
            if m in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
                h, w = struct.unpack('>HH', data[1:5])
                return [w, h]


def ply(path):
    lines = path.read_text().splitlines()
    stop = lines.index('end_header')
    head = lines[:stop+1]
    if 'format ascii 1.0' not in head:
        raise ValueError('ASCII PLY만 지원')
    nv = int(next(x.split()[-1] for x in head if x.startswith('element vertex ')))
    nf = int(next(x.split()[-1] for x in head if x.startswith('element face ')))
    vertices = [list(map(float, x.split()[:3])) for x in lines[stop+1:stop+1+nv]]
    faces = []
    for line in lines[stop+1+nv:stop+1+nv+nf]:
        v = list(map(int, line.split()))
        if len(v) != v[0]+1 or v[0] != 3 or any(i < 0 or i >= nv for i in v[1:]):
            raise ValueError('삼각면/인덱스 형식 오류')
        faces.append(v[1:])
    return vertices, faces, head


def det3(R):
    a,b,c=R[0];d,e,f=R[1];g,h,i=R[2]
    return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)


def transform(v, R, t):
    return [sum(R[i][j]*v[j] for j in range(3))+t[i] for i in range(3)]


def main():
    gt = json.loads((SOURCE/'scene_gt_obj04.json').read_text())
    cameras = json.loads((SOURCE/'scene_camera_obj04.json').read_text())
    params = json.loads((SOURCE/'camera_params.json').read_text())
    paths = {int(p.stem.split('_')[-1]): p for p in (SOURCE/'poses_obj04').glob('pose_*.npy')}
    image_sets = {name: {int(p.stem): p for p in (SOURCE/name).glob('*.jpg')}
                  for name in ('image', 'image_undistorted')}
    ids = set(paths)
    mismatches = {name: {'missing': sorted(ids-set(mapping)), 'extra': sorted(set(mapping)-ids)}
                  for name,mapping in {**image_sets, 'gt': {int(k):v for k,v in gt.items()},
                                       'camera': {int(k):v for k,v in cameras.items()}}.items()}
    verts, faces, header = ply(SOURCE/'obj_000004.ply')
    lo = [min(v[a] for v in verts) for a in range(3)]
    hi = [max(v[a] for v in verts) for a in range(3)]
    ext = [hi[a]-lo[a] for a in range(3)]
    edges = {}
    degenerate = 0
    for face in faces:
        a,b,c = [verts[i] for i in face]
        u=[b[j]-a[j] for j in range(3)];v=[c[j]-a[j] for j in range(3)]
        cross=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
        degenerate += sum(x*x for x in cross) < 1e-20
        for k in range(3):
            edge=tuple(sorted((face[k],face[(k+1)%3])))
            edges[edge]=edges.get(edge,0)+1
    # 정확한 vertex-pair 최대 거리. 1439점, CPU 표준 수치 계산만 수행.
    diameter = math.sqrt(max(sum((a[j]-b[j])**2 for j in range(3))
                             for n,a in enumerate(verts) for b in verts[n+1:]))
    rows=[]
    for frame,path in sorted(paths.items()):
        P,h = npy_matrix(path)
        entries=gt[str(frame)]
        if len(entries)!=1 or entries[0]['obj_id']!=4:
            raise ValueError(f'{frame}: obj_id/인스턴스 개수 오류')
        item=entries[0]
        R=[P[i][:3] for i in range(3)]
        t=[P[i][3] for i in range(3)]
        rj=[item['cam_R_m2c'][i*3:i*3+3] for i in range(3)]
        tj=item['cam_t_m2c']
        K=params['camera_matrix']
        flat=[x for row in K for x in row]
        gram=max(abs(sum(R[k][i]*R[k][j] for k in range(3))-(i==j))
                 for i in range(3) for j in range(3))
        xyz=[transform(v,rj,tj) for v in verts]
        rows.append({'frame':frame,'npy_header':h,'finite':all(math.isfinite(x) for row in P for x in row),
                     'bottom_row_max_error':max(abs(P[3][i]-[0,0,0,1][i]) for i in range(4)),
                     'rotation_max_diff':max(abs(R[i][j]-rj[i][j]) for i in range(3) for j in range(3)),
                     'rotation_orthogonality_max_error':gram,'rotation_determinant':det3(R),
                     'translation_npy':t,'translation_json':tj,
                     'translation_max_diffs':{str(s):max(abs(t[i]*s-tj[i]) for i in range(3)) for s in (1.,1000.,.001)},
                     'camera_K_max_diff':max(abs(a-b) for a,b in zip(flat,cameras[str(frame)]['cam_K'])),
                     'camera_K_length':len(cameras[str(frame)]['cam_K']),
                     'depth_scale':cameras[str(frame)].get('depth_scale'),
                     'image_sizes':{name:jpeg_size(mapping[frame]) for name,mapping in image_sets.items()},
                     'cad_camera_z_range_native_units':[min(v[2] for v in xyz),max(v[2] for v in xyz)]})
    files=[p for p in SOURCE.rglob('*') if p.is_file() and not p.is_symlink()]
    manifest=[{'path':str(p.relative_to(SOURCE)), 'bytes':p.stat().st_size,
               'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
    result={'source':str(SOURCE),'count':len(rows),'frame_ids':sorted(ids),
            'frame_mismatches':mismatches,'camera_params':params,
            'cad':{'vertices':len(verts),'faces':len(faces),'header':header,'min_xyz':lo,'max_xyz':hi,
                   'size_xyz':ext,'aabb_center':[(a+b)/2 for a,b in zip(lo,hi)],
                   'vertex_diameter':diameter,'degenerate_faces':degenerate,
                   'boundary_edges':sum(v==1 for v in edges.values()),
                   'nonmanifold_edges':sum(v>2 for v in edges.values()),
                   'unit_explicit_in_header':False},'frames':rows,'source_manifest':manifest}
    out=OUT/'numeric_results.json'
    with out.open('x',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('frames','source_manifest')},ensure_ascii=False,indent=2))
    print('SUMMARY',json.dumps({'R_max_diff':max(r['rotation_max_diff'] for r in rows),
        't_max_diffs':{str(s):max(r['translation_max_diffs'][str(s)] for r in rows) for s in (1.,1000.,.001)},
        'orthogonality_max_error':max(r['rotation_orthogonality_max_error'] for r in rows),
        'determinant_range':[min(r['rotation_determinant'] for r in rows),max(r['rotation_determinant'] for r in rows)],
        'finite_all':all(r['finite'] for r in rows),'bottom_max_error':max(r['bottom_row_max_error'] for r in rows),
        'K_max_diff':max(r['camera_K_max_diff'] for r in rows),
        'image_sizes':sorted({tuple(sz) for r in rows for sz in r['image_sizes'].values()}),
        't_json_ranges':[[min(r['translation_json'][a] for r in rows),max(r['translation_json'][a] for r in rows)] for a in range(3)],
        'cad_camera_z_global':[min(r['cad_camera_z_range_native_units'][0] for r in rows),max(r['cad_camera_z_range_native_units'][1] for r in rows)],
        'source_files':len(files),'source_bytes':sum(p.stat().st_size for p in files)},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
