from transformers import AutoTokenizer
from srctwo.data.utils import test_data,distinguish_eval
import json
model_path = r"D:\vscode--llm\my-llm-training\srctwo\models\Qwen2.5-7B"


eval_datas=distinguish_eval(test_data)
print(eval_datas[0])
true_entities=[]
true_entities=[json.loads(eval_data['entities']) for eval_data in eval_datas]
print(true_entities[1])

from collections import defaultdict
# true_entities = [
#     [["口苦","临床表现"]],
#     [["复方蜥蜴散","方剂"], ["三七","中药"]],
#     ...
# ]

# pred_entities = [
#     [["口苦","临床表现"]],
#     [["复方蜥蜴散","方剂"], ["半枝莲","中药"]],
#     ...
# ]

def compute_f1(true_entities, pred_entities):
    # 统计 TP, FP, FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for true, pred in zip(true_entities, pred_entities):
        #true=[['复方蜥蜴散', '方剂'], ['三七', '中药']]  pred=[['复方蜥蜴散', '方剂'], ['半枝莲', '中药']]
        true_set = {(t, c) for t, c in true}
        #{('复方蜥蜴散', '方剂'), ('三七', '中药')}
        pred_set = {(t, c) for t, c in pred}
        #转成集合才能求交集

# 统计 TP
        for ent in (true_set & pred_set):#必须完全一样，否则就是空的
            tp[ent[1]] += 1   # ent[1] 是类别

        # 统计 FP
        for ent in (pred_set - true_set):#只保留 pred 中有，但 true 中没有的元素。pred_set - true_set如果是 {('半枝莲', '中药')}和{('一枝莲', '方剂')}则保留{('半枝莲', '中药')}
            fp[ent[1]] += 1
# 虽然“半枝莲”相同，但元组不同，所以：
# 它们被视为两个不同的实体
# 差集也会把它们区分开
        # 统计 FN
        for ent in (true_set - pred_set):
            fn[ent[1]] += 1

    # 计算 F1
    for cls in tp.keys() | fp.keys() | fn.keys():
        precision = tp[cls] / (tp[cls] + fp[cls] + 1e-8)#1 × 10⁻⁸
        recall    = tp[cls] / (tp[cls] + fn[cls] + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"类别：{cls}")
        print(f"  Precision={precision:.4f}")
        print(f"  Recall={recall:.4f}")
        print(f"  F1={f1:.4f}\n")

        
    #     # 每类 TP / FN / FP
    #     for ent in true_set:
    #         if ent in pred_set:
    #             tp[ent[1]] += 1
    #         else:
    #             fn[ent[1]] += 1

    #     for ent in pred_set:
    #         if ent not in true_set:
    #             fp[ent[1]] += 1

    # # 计算每个类别 F1
    # results = {}
    # for cat in set(list(tp.keys()) + list(fp.keys()) + list(fn.keys())):
    #     p = tp[cat] / (tp[cat] + fp[cat]) if tp[cat] + fp[cat] > 0 else 0
    #     r = tp[cat] / (tp[cat] + fn[cat]) if tp[cat] + fn[cat] > 0 else 0
    #     f1 = 2*p*r / (p+r) if p+r > 0 else 0
    #     results[cat] = {"precision": p, "recall": r, "f1": f1}

    # # 计算 overall micro F1
    # TP = sum(tp.values())
    # FP = sum(fp.values())
    # FN = sum(fn.values())
    # P = TP / (TP + FP) if TP + FP > 0 else 0
    # R = TP / (TP + FN) if TP + FN > 0 else 0
    # overall_f1 = 2*P*R / (P+R) if P+R > 0 else 0

    # return results, overall_f1
