# get ./core/data/answer_dict.json
# if the appear times of the word in the answer word list <= 8, it will be poped
# finally, the length of answer word list = 3129

import sys
# sys.path.append('../')
from core.data.ans_punct import prep_ans
import json

DATASET_PATH = '../../datasets-vqa/vqa/'

ANSWER_PATH = {
    'train': DATASET_PATH + 'v2_mscoco_train2014_annotations.json',
    'val': DATASET_PATH + 'v2_mscoco_val2014_annotations.json',
    'vg': DATASET_PATH + 'VG_annotations.json',
}

x = json.load(open(ANSWER_PATH['train'], 'r'))
y = json.load(open(ANSWER_PATH['val'], 'r'))

# Loading answer word list
stat_ans_list = \
    json.load(open(ANSWER_PATH['train'], 'r'))['annotations'] + json.load(open(ANSWER_PATH['val'], 'r'))['annotations']


def ans_stat(stat_ans_list):
    ans_to_ix = { }
    ix_to_ans = {}
    ans_freq_dict = {}

    # 统计每个答案出现的次数
    for ans in stat_ans_list:
        ans_proc = prep_ans(ans['multiple_choice_answer'])
        if ans_proc not in ans_freq_dict:
            ans_freq_dict[ans_proc] = 1
        else:
            ans_freq_dict[ans_proc] += 1

    # 如果某个答案出现次数<=8就pop掉
    ans_freq_filter = ans_freq_dict.copy()
    for ans in ans_freq_dict:
        if ans_freq_dict[ans] <= 8:
            ans_freq_filter.pop(ans)

    # ix_to_ans[0] = 'net'
    # ans_to_ix['net'] = 0
    for ans in ans_freq_filter:
        ix_to_ans[ans_to_ix.__len__()] = ans
        ans_to_ix[ans] = ans_to_ix.__len__()

    return ans_to_ix, ix_to_ans

ans_to_ix, ix_to_ans = ans_stat(stat_ans_list)
# ans_to_ix.__len__() = 3129
json.dump([ans_to_ix, ix_to_ans], open('../core/data/answer_dict.json', 'w'))
