import json
import os
from tqdm import tqdm
import random


class Args:
    para_file_name = "para_pairs_v2.json"
    work_dir = "./output"
    save_dir = "../data"
    templates = {
        2: "{}的{}",
        3: "{}的{}中的{}",
        4: "{}的{}中，{}的{}"
    }
    output_file_name = "instruction_para_v2.json"
    interval_len = 1000


def build_positive_instruction(pairs):
    "构建指令：用第一个tag来替换实体名字"
    for pair in tqdm(pairs, desc="building positive instruction"):
        title_levels = len(pair["titles"])
        instruction = Args.templates[title_levels].format(pair["tag"][0], *pair["titles"][1:])
        pair["instructions"] = instruction


elements = [
    (['五一东街', '城市设计', '现代建城规制', None], 3),
    (['五一东街', '城市规划', '古代建城规制', None], 3),
    (['五一东街', '历史背景', '金城汤池', None], 3),
    (['五一东街', '历史背景', '太原皮鞋厂', None], 3),
    (['五一东街', '历史渊源', None, None], 2),
    (['李白', '生平', None, None], 2),
    (['李白', '诗词', "唐诗", None], 3),
    (['李白', '晚年', '文化', None], 3),
    (['李白', '晚年', '交友', None], 3),
    (['李白', '经历', '生平', None], 3)
]


def title_pad(title):
    valid_lens = len(title)
    pad_lens = 4 - valid_lens
    return title + [None] * pad_lens, valid_lens


def right_search(candidates, title):
    res = []
    init_title = title[0][:]

    def downward_expand(candidates, title):
        """从第0个index向下索引，直到找不到一样的为止"""
        deepth = len(title)
        if deepth == 0:
            return

        i = 0
        while i < len(candidates) and candidates[i][0][:deepth] == title:
            if candidates[i][0] != init_title:
                res.append(candidates[i])
            i += 1

        if res == []:
            downward_expand(candidates, title[:-1])

    downward_expand(candidates, title[0])
    return res


def left_search(candidates, title):
    res = []
    init_title = title[0][:]

    def up_expand(candidates, title):
        """从第0个index向下索引，直到找不到一样的为止"""
        deepth = len(title)
        if deepth == 0:
            return

        i = len(candidates) - 1
        while i >= 0 and candidates[i][0][:deepth] == title:
            if candidates[i][0] != init_title:
                res.append(candidates[i])
            i -= 1

        if res == []:
            up_expand(candidates, title[:-1])

    up_expand(candidates, title[0])
    return res


def build_instructions(pairs):
    """1、构建指令集
    2、按照tag构建指令树,供找不到同实体指令的时候用"""
    instructions = []
    instructions_tree = {}
    for pair in tqdm(pairs, desc="building tree of instruction"):
        instruction = pair["titles"].copy()
        instruction[0] = pair["tag"][0]
        pad_instruction, valid_lens = title_pad(instruction)
        instructions.append((pad_instruction, valid_lens))
        if pair['tag'][0] not in instructions_tree:
            instructions_tree[pair['tag'][0]] = []
        instructions_tree[pair['tag'][0]].append(instruction)
    return instructions, instructions_tree


def build_negative_sample(pairs, instructions, instructions_tree):
    "根据指令集合，在每一个指令的上下进行搜索，选择语义最近的，构建困难负例"
    for i in tqdm(range(len(instructions)), desc="searching negative candidates"):
        left_title, right_title = [], []
        if i > 0:
            left_title = left_search(instructions[i - Args.interval_len:i], instructions[i])
        if i < len(instructions) - 1:
            right_title = right_search(instructions[i + 1: i + 1 + Args.interval_len], instructions[i])
        negative_candidates = left_title + right_title
        # 下面抽样的过程中没有去掉hard negative candidates中的重复项，是希望能有一个按出现频率的抽样结果
        if negative_candidates != []:
            negative = random.choice(negative_candidates)
        else:
            negative = random.choice(instructions_tree[instructions[i][0][0]])
            negative = (negative, len(negative))
        pairs[i]["negative"] = Args.templates[negative[1]].format(*negative[0])


if __name__ == "__main__":
    print(left_search(elements[:4], elements[4]))
    print(right_search(elements[6:], elements[5]))
    pairs = json.load(open(os.path.join(Args.work_dir, Args.para_file_name), "r"))
    build_positive_instruction(pairs)
    instructions, instructions_tree = build_instructions(pairs)
    build_negative_sample(pairs, instructions, instructions_tree)
    with open(os.path.join(Args.save_dir, Args.output_file_name), "w") as f:
        json.dump(pairs, f, indent=4)
