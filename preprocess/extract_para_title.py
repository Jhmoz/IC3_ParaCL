import json
import os
from tqdm import tqdm

class Args:
    para_file_name = "para_dict_v2.json"
    tag_file_name = "tag_summary_v2.json"
    work_dir = "./output"
    save_dir = "./output"
    output_file = "para_pairs_v2.json"

paras = json.load(open(os.path.join(Args.work_dir, Args.para_file_name), "r"))
tags = json.load(open(os.path.join(Args.work_dir, Args.tag_file_name), "r"))
os.makedirs(Args.save_dir, exist_ok=True)

def extract_tags(per_tag):
    res = []

    def find_tags(tag):
        if 'tagName' in tag:
            res.append(tag['tagName'])
            return

        if isinstance(tag, dict):
            for k in tag:
                find_tags(tag[k])
        else:
            for subtag in tag:
                find_tags(subtag)

    find_tags(per_tag)
    return res


def find_all_title(para):
    res = []
    level = []

    def callback_find(para):
        if isinstance(para, str):
            if len(para) < 512:
                res.append(
                    {
                        "title": level[:],
                        "paragraph": para
                    }
                )
            return

        elif isinstance(para, dict):
            for title in para:
                level.append(title)
                callback_find(para[title])
                level.pop()
        else:
            return

    callback_find(para)
    return res


def batch_extract(paras):
    res = []
    for lemma_id in tqdm(paras, desc="extracting titles"):
        if tags[lemma_id]['tag'] == 'None':
            continue
        cur_pairs = find_all_title(paras[lemma_id]["paragragh"])
        for cur_pair in cur_pairs:
            cur_title = cur_pair["title"]
            if len(cur_title) == 1 and cur_title == paras[lemma_id]['name']:
                continue
            cur_title.insert(0, paras[lemma_id]["name"])
            res.append(
                {
                    "titles": cur_title,
                    "paragraphs": cur_pair["paragraph"],
                    "tag": extract_tags(tags[lemma_id]["tag"])
                }
            )
    return res


def save_files(to_save: set):
    print("title counts: {}".format(len(to_save)))
    with open(os.path.join(Args.save_dir, Args.output_file), "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=4)


if __name__ == "__main__":
    pairs = batch_extract(paras)
    save_files(pairs)
