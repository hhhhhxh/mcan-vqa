import json

VG_que_path = '../datasets-vqa/vqa/VG_questions.json'
VG_que = json.load(open(VG_que_path, 'r'))
VG_anno_path = '../datasets-vqa/vqa/VG_annotations.json'
VG_anno = json.load(open(VG_anno_path, 'r'))
pass

# VG_questions.json: {'image_id':  33554, 'question': 'What is the church made of?', 'question_id': 'VG_135804'}