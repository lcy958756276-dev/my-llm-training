from typing import Optional

from datasets import IterableDataset,load_dataset

# def load_medical_dataset(path):
#     with open(path,"r",encoding="utf-8") as f:
#         lines = f.read().splitlines()
#     for id,line in enumerate(lines):
#         print(f"{line}")


#load_medical_dataset("D:\\vscode--llm\\stage3_data\\llm_data\\medical.train")


from typing import Optional

from datasets import IterableDataset,load_dataset
# from torch.utils.data import Dataset
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset
import json


categories = set()
num=0
class PeopleDaily(TorchDataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
        # self.dataset = Dataset.from_dict(self.data)
        self.dataset =  HFDataset.from_list(self.data)
    
    def load_data(self, data_file):
        #Data = {}
        global num
        Data=[]
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence, labels = '', []
                for i, item in enumerate(line.split('\n')):
                    char, tag = item.split(' ')
                    sentence += char
                    if tag.startswith('B'):
                        # labels.append([i, i, char, tag[2:]]) # Remove the B- or I-
                        labels.append([char, tag[2:]]) # Remove the B- or I-
                        categories.add(tag[2:])
                    elif tag.startswith('I'):
                        #labels[-1][1] = i
                        #labels[-1][2] += char
                        labels[-1][0]+=char
                # Data[idx] = {
                #     'sentence': sentence, 
                #     'labels': labels
                # }
#                 {
#   0: {"sentence": s1, "labels": l1},
#   1: {"sentence": s2, "labels": l2},
#   2: {"sentence": s3, "labels": l3}
# }
                Data.append({
                    'sentence': sentence, 
                     'labels': labels
                })
                num+=1

        return Data

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     return self.data[idx]
    def to_hf(self):
        return self.dataset

    def __getitem__(self, idx):
    # 如果是 slice，比如 [:10]
        if isinstance(idx, slice):
        # 返回一个列表（或新的 Dataset）
            return [self.data[i] for i in range(*idx.indices(len(self.data)))]

    # 普通整数 index
        return self.data[idx]


train_data = PeopleDaily('D:\\vscode--llm\\stage3_data\\llm_data\\medical.train').to_hf()
print(f"{num}")
# valid_data = PeopleDaily('D:\\vscode--llm\\stage3_data\\llm_data\\medical.dev').to_hf()
# test_data = PeopleDaily('D:\\vscode--llm\\stage3_data\\llm_data\\medical.test').to_hf()

# print(type(train_data[0]))---->dict
# print(train_data[1])
# print(train_data[2])
# print(train_data[4])
# print(train_data[5])

# {'sentence': '现头昏口苦', 'labels': [['口苦', '临床表现']]}
# {'sentence': '目的观察复方丁香开胃贴外敷神阙穴治疗慢性心功能不全伴功能性消化不良的临床疗效', 'labels': [['复方丁香开胃贴', '中医治疗'], ['心功能不全伴功能性消化不良', '西医诊断']]}
# {'sentence': '舒肝和胃消痞汤；功能性消化不良', 'labels': [['功能性消化不良', '西医诊断']]}
# {'sentence': '治疗组采用复方蜥蜴散不同微粒组合剂（密点麻蜥、炙黄芪、焦乌梅、炒白芍、三七、半枝莲等）治疗', 'labels': [['复方蜥蜴散', '方剂'], ['密点麻蜥', '中药'], ['炙黄芪', '中药'], ['焦乌梅', '中药'], ['炒
# 白芍', '中药'], ['三七', '中药'], ['半枝莲', '中药']]}
# {'sentence': '经检查诊断为“缩窄性心包炎”', 'labels': [['缩窄性心包炎', '西医诊断']]}


SYSTEM_prompt=(
    "你是一个智能助手。给定一句话，你需要抽取其中的医学实体，并判断其类别，"
    "输出格式为[[实体，类别]...]"
    )


#分别提取sentence和labels
def distinguish(data):
    data=data.map(lambda x:{
        'prompt':[
            {'role':'system','content':SYSTEM_prompt},
            {'role':'user','content':x["sentence"]},
            {'role':'assistant','content':json.dumps(x['labels'],ensure_ascii=False)}
        ]
    },
    remove_columns=data.column_names  # ← 关键！
    )
    return data

def distinguish_eval(data):
    data=data.map(lambda x:{
        'text':x['sentence'],
        'entities':json.dumps(x['labels'],ensure_ascii=False)
    },
    remove_columns=data.column_names  # ← 关键！
    )
    return data