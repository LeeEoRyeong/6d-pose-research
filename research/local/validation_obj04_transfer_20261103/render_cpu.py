"""승인 후에만 실행: 표준 라이브러리 CPU z-buffer + PNG/SVG 오버레이."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

from audit_numeric import SOURCE, OUT, ply, transform


def png_rgba(width, height, pixels):
    def chunk(name, data):
        return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
    raw=b''.join(b'\0'+pixels[y*width*4:(y+1)*width*4] for y in range(height))
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',width,height,8,6,0,0,0))
            +chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))


def project(point, K, distortion):
    x,y,z=point
    if z<=0:
        raise ValueError('카메라 뒤의 CAD 꼭짓점: clipping 미지원')
    x/=z;y/=z
    if distortion is not None:
        k1,k2,p1,p2,k3=distortion
        r2=x*x+y*y;factor=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        x,y=x*factor+2*p1*x*y+p2*(r2+2*x*x),y*factor+p1*(r2+2*y*y)+2*p2*x*y
    return [K[0][0]*x+K[0][1]*y+K[0][2],K[1][1]*y+K[1][2],z]


def raster(points, faces, width, height):
    # 순수 CPU 래스터화. 모든 삼각면을 그려 winding 방향에 의존하지 않는다.
    depth=[math.inf]*(width*height)
    pixels=bytearray(width*height*4)
    for face in faces:
        a,b,c=[points[i] for i in face]
        ax,ay,az=a;bx,by,bz=b;cx,cy,cz=c
        den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(den)<1e-10:
            continue
        x0=max(0,math.floor(min(ax,bx,cx)));x1=min(width-1,math.ceil(max(ax,bx,cx)))
        y0=max(0,math.floor(min(ay,by,cy)));y1=min(height-1,math.ceil(max(ay,by,cy)))
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):
                u=((by-cy)*(x-cx)+(cx-bx)*(y-cy))/den
                v=((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/den
                w=1-u-v
                if min(u,v,w)<-1e-8:
                    continue
                z=1/(u/az+v/bz+w/cz)
                at=y*width+x
                if z<depth[at]:
                    depth[at]=z
                    pixels[at*4:at*4+4]=bytes((0,230,100,110))
    coverage=bytearray(pixels[i+3]>0 for i in range(0,len(pixels),4))
    for y in range(1,height-1):
        for x in range(1,width-1):
            i=y*width+x
            if coverage[i] and not all(coverage[k] for k in (i-1,i+1,i-width,i+width)):
                pixels[i*4:i*4+4]=bytes((0,255,100,255))
    return png_rgba(width,height,pixels),sum(coverage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print('미실행. 재투영 실행 승인 후 --run을 지정하세요.');return
    numeric=json.loads((OUT/'numeric_results.json').read_text())
    # 원본 manifest는 읽기 전용 확인에만 사용한다.
    for item in numeric['source_manifest']:
        path=SOURCE/item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'수치 검사 이후 원본 변경: {path}')
    dest=OUT/'reprojection_cpu'
    dest.mkdir(exist_ok=False)
    verts,faces,_=ply(SOURCE/'obj_000004.ply')
    gt=json.loads((SOURCE/'scene_gt_obj04.json').read_text())
    cams=json.loads((SOURCE/'scene_camera_obj04.json').read_text())
    params=numeric['camera_params']
    width,height=params['image_width'],params['image_height']
    records=[]; gallery=['<!doctype html><meta charset="utf-8"><title>obj04 CPU 재투영</title>',
        '<h1>obj04 CPU 재투영</h1><p>녹색: CAD, 적/녹/청: +X/+Y/+Z. '
        'raw는 K+D, undistorted는 같은 K 사용 가설. 시각 확인용이며 Pose 정확도 점수가 아닙니다.</p>']
    for frame in numeric['frame_ids']:
        item=gt[str(frame)][0]
        R=[item['cam_R_m2c'][i*3:i*3+3] for i in range(3)];t=item['cam_t_m2c']
        flat=cams[str(frame)]['cam_K'];K=[flat[i*3:i*3+3] for i in range(3)]
        xyz=[transform(v,R,t) for v in verts]
        gallery.append(f'<h2>frame {frame:06d}</h2>')
        for mode,folder,D in [('raw_K_D','image',params['distortion_coefficients']),
                              ('undistorted_sameK_HYPOTHESIS','image_undistorted',None)]:
            print(f'CPU 렌더링 {frame:06d} {mode}',flush=True)
            projected=[project(v,K,D) for v in xyz]
            png,covered=raster(projected,faces,width,height)
            stem=f'{frame:06d}_{mode}'
            (dest/f'{stem}_cad.png').write_bytes(png)
            original=SOURCE/folder/f'{frame:06d}.jpg'
            jpeg=base64.b64encode(original.read_bytes()).decode('ascii')
            overlay=base64.b64encode(png).decode('ascii')
            axes=[project(transform(p,R,t),K,D) for p in ([0,0,0],[30,0,0],[0,30,0],[0,0,30])]
            svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 f'<image width="{width}" height="{height}" href="data:image/jpeg;base64,{jpeg}"/>',
                 f'<image width="{width}" height="{height}" href="data:image/png;base64,{overlay}"/>']
            for endpoint,color,label in zip(axes[1:],('red','lime','blue'),('X','Y','Z')):
                svg.append(f'<line x1="{axes[0][0]}" y1="{axes[0][1]}" x2="{endpoint[0]}" y2="{endpoint[1]}" stroke="{color}" stroke-width="2"/>')
                svg.append(f'<text x="{endpoint[0]}" y="{endpoint[1]}" fill="{color}">{label}</text>')
            svg.append(f'<text x="10" y="24" fill="yellow">{frame:06d} {mode}</text></svg>')
            (dest/f'{stem}.svg').write_text('\n'.join(svg),encoding='utf8')
            gallery.append(f'<figure style="display:inline-block;width:48%"><figcaption>{mode}</figcaption><img style="width:100%" src="{stem}.svg"></figure>')
            records.append({'frame':frame,'mode':mode,'visible_pixels':covered,
                            'projected_vertex_bbox':[min(p[0] for p in projected),min(p[1] for p in projected),
                                                     max(p[0] for p in projected),max(p[1] for p in projected)]})
    (dest/'index.html').write_text('\n'.join(gallery),encoding='utf8')
    (dest/'projection_metadata.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
    print(f'출력: {dest}')


if __name__=='__main__':
    main()
