import json
import os
from extract_para_title import extract_tags
from tqdm import tqdm

class Args:
    work_dir = "../baike"
    save_dir = "./output"
    keywords_file = "query2question.txt"
    tag_file_name = "tag&summary_dict.json"
    num_questions = 1041326


os.makedirs(Args.save_dir, exist_ok=True)
tags = json.load(open(os.path.join(Args.work_dir, Args.tag_file_name), "r"))

def build_name2tag(tags: dict):
    res={}
    for lemma_id in tqdm(tags,desc="building name2tag"):
        if lemma_id not in res and tags[lemma_id]['tag']!='None':
            res[tags[lemma_id]["name"]] = tags[lemma_id]["tag"]
        else:
            res[tags[lemma_id]['name']] = None
    return res

name2tag = build_name2tag(tags)

def build_keyword2questions():
    res = {}
    with open(os.path.join(Args.work_dir, Args.keywords_file), "r") as f:
        with tqdm(total=Args.num_questions, desc="Reading keywords file", unit="line") as pbar:
            for line in f:
                splits = line.strip().split("\t")[1:]
                keywords = splits[0].split()
                lemma_name = " ".join(keywords[:-1])
                lemma_attribution = keywords[-1]
                if lemma_name not in res:
                    res[lemma_name] = {}
                    if lemma_name in name2tag:
                        if name2tag[lemma_name] is not None:
                            res[lemma_name]["tag"] = extract_tags(name2tag[lemma_name])

                if lemma_attribution not in res[lemma_name]:
                    res[lemma_name][lemma_attribution] = []
                res[lemma_name][lemma_attribution].append(splits[-1])
                pbar.update(1)
    return res


if __name__ == "__main__":
    keyword2questions = build_keyword2questions()
    with open(os.path.join(Args.save_dir, "keywords2querys.json"), "w", encoding="utf-8") as f:
        json.dump(keyword2questions, f, indent=4)
