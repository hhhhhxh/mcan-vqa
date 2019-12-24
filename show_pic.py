from PIL import Image
import numpy as np
import json

npz_path = '../datasets-vqa/coco_extract/val2014/COCO_val2014_'
img_path = '000000000042.jpg.npz'
annotation_file = '../datasets-vqa/vqa/v2_mscoco_val2014_annotations.json'
question_file = '../datasets-vqa/vqa/v2_OpenEnded_mscoco_val2014_questions.json'

img = np.load(npz_path + img_path)
print(img['x'])
print(img['image_w'])
print(img['image_h'])
print(img['bbox'])
print(img['num_bbox'])

annotation = json.load(open(annotation_file, 'r'))
question = json.load(open(question_file, 'r'))

