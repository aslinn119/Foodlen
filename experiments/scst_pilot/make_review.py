from experiment import *
from PIL import ImageDraw
rows=read(ROOT/'results/candidates.jsonl')[100:106]
sheet=Image.new('RGB',(900,600),'white')
for i,r in enumerate(rows):
    with Image.open(BASE/'Food500Cap_ready'/r['file_name']) as im:
        im.thumbnail((290,260));x=(i%3)*300;y=(i//3)*300;sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x,y+265),str(i+1),fill='black')
sheet.save(ROOT/'results/audit_images.jpg')
for i,r in enumerate(rows):print(i+1,r['image_id'],'\nREF',r['reference'],'\nGREEDY',r['greedy'],'\nBEAM',r['candidates'][0]['caption'])
