from PIL import Image
import json

anno_file = './data/v-coco/annotations/test_vcoco.json'
with open(anno_file, 'r') as f:
    annotations = json.load(f)
for i in range(len(annotations)):
    if i%50 == 0:
        print(i)
    img_anno = annotations[i]
    img = Image.open('./data/v-coco/images/val2014/'+ img_anno['file_name']).convert('RGB')
    img.save('./data/v-coco/images/val2014_/'+ img_anno['file_name'])