"""기존 시스템 Pillow만 사용해 생성물의 CPU 시각 검토 PNG를 추가한다."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/RESEARCH_DATA/HCCEPose_obj04_transfer')


def main():
    dest=ROOT/'review_previews'
    dest.mkdir(exist_ok=False)
    numeric=json.loads((ROOT/'numeric_results.json').read_text())
    for item in numeric['source_manifest']:
        if hashlib.sha256((SOURCE/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('원본 해시 변경: '+item['path'])
    records=json.loads((ROOT/'reprojection_cpu/projection_metadata.json').read_text())
    lookup={(r['frame'],r['mode']):r for r in records}
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    rows={}; sizes={}
    for frame in numeric['frame_ids']:
        full=Image.new('RGB',(2560,752),'#202020')
        detail=Image.new('RGB',(1600,432),'#202020')
        for index,(mode,folder) in enumerate((('raw_K_D','image'),('undistorted_sameK_HYPOTHESIS','image_undistorted'))):
            stem=f'{frame:06d}_{mode}'
            # load()로 JPEG의 실제 디코딩까지 검사한다.
            with Image.open(SOURCE/folder/f'{frame:06d}.jpg') as image:
                image.load(); original=image.convert('RGBA')
                sizes[f'{folder}/{frame:06d}.jpg']=list(image.size)
            with Image.open(ROOT/'reprojection_cpu'/f'{stem}_cad.png') as cad:
                combined=Image.alpha_composite(original,cad.convert('RGBA'))
            svg=ET.parse(ROOT/'reprojection_cpu'/f'{stem}.svg').getroot()
            draw=ImageDraw.Draw(combined)
            for line in svg.findall('{http://www.w3.org/2000/svg}line'):
                a=line.attrib
                draw.line([(float(a['x1']),float(a['y1'])),(float(a['x2']),float(a['y2']))],fill=a['stroke'],width=2)
            for text in svg.findall('{http://www.w3.org/2000/svg}text'):
                a=text.attrib
                if text.text in ('X','Y','Z'):
                    draw.text((float(a['x']),float(a['y'])),text.text,fill=a['fill'],font=font)
            full.paste(combined.convert('RGB'),(index*1280,32))
            ImageDraw.Draw(full).text((index*1280+8,5),f'{frame:06d} {mode}',fill='white',font=font)
            x0,y0,x1,y1=lookup[(frame,mode)]['projected_vertex_bbox']
            side=max(x1-x0,y1-y0)+120
            cx=(x0+x1)/2;cy=(y0+y1)/2
            box=(round(cx-side/2),round(cy-side/2),round(cx+side/2),round(cy+side/2))
            for sub,img in enumerate((original,combined)):
                crop=img.crop(box).convert('RGB').resize((400,400),Image.Resampling.LANCZOS)
                detail.paste(crop,((index*2+sub)*400,32))
                label=f'{frame:06d} '+('RAW' if index==0 else 'UND same-K?')+(' original' if sub==0 else ' + CAD')
                ImageDraw.Draw(detail).text(((index*2+sub)*400+5,5),label,fill='white',font=font)
        full.save(dest/f'{frame:06d}_full.png')
        detail.save(dest/f'{frame:06d}_detail.png')
        rows[frame]=detail
    priority=[26,51,56,92,126]
    order=priority+[f for f in numeric['frame_ids'] if f not in priority]
    for start in range(0,len(order),3):
        frames=order[start:start+3]
        sheet=Image.new('RGB',(1600,432*len(frames)),'#202020')
        for row,frame in enumerate(frames):sheet.paste(rows[frame],(0,row*432))
        sheet.save(dest/f'sheet_{start//3+1:02d}.png')
    with (dest/'decoded_images.json').open('x',encoding='utf8') as f:
        json.dump(sizes,f,indent=2)
    print(f'검토 PNG 54개, 3행 contact sheet 9개, 디코딩 메타데이터: {dest}')


if __name__=='__main__':main()
