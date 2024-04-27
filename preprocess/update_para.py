import os
import json
from tqdm import tqdm


class Files:
    work_dir = "../baike"
    old = "para_dict_fixed.json"
    new = "new_para.jsonl"
    output_dir = "./output"



def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        res = json.load(f)
    return res


def read_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            yield line


def saving_json(filepath,data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def combine(old, gener):
    unmatch = 0
    unmatch_list = []
    for line in tqdm(gener):
        lemma_id = line['lemma_id']
        if lemma_id not in old:
            old[lemma_id] = line
        else:
            name = line["name"]
            # if name in old[lemma_id]["name"] or old[lemma_id]["name"] in name:
            para_titles = list(line["paragragh"].keys())
            for para_title in para_titles:
                if para_title not in old[lemma_id]["paragragh"]:
                    old[lemma_id]["paragragh"][para_title] = line["paragragh"][para_title]

    saving_json(os.path.join(Files.output_dir,"para_dict_v2.json"),old)
    # print(unmatch_list[:20])
    # print(unmatch)



if __name__ == "__main__":
    old = read_json(os.path.join(Files.work_dir, Files.old))
    gener = read_jsonl(os.path.join(Files.work_dir, Files.new))
    combine(old, gener)
