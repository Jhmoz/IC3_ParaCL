import os
import json
from tqdm import tqdm


class Files:
    work_dir = "../baike"
    old = "info_dict.json"
    new = "new_info.json"
    output_dir = "./output"


def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        res = json.load(f)
    return res

def saving_json(filepath,data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def combine(old, new):
    # unmatch = 0
    # unmatch_list = []
    for lemma_id in tqdm(new):
        if new[lemma_id]['basicInfo']=='None':
            continue
        if lemma_id not in old:
            old[lemma_id] = new[lemma_id]
        else:
            # if name in old[lemma_id]["name"] or old[lemma_id]["name"] in name:
            info_titles = list(new[lemma_id]['basicInfo'].keys())
            if old[lemma_id]['basicInfo'] != "None":
                for info_title in info_titles:
                    if info_title not in old[lemma_id]['basicInfo']:
                        old[lemma_id]["basicInfo"][info_title] = new[lemma_id]['basicInfo'][info_title]
            else:
                old[lemma_id]['basicInfo'] = new[lemma_id]['basicInfo']
    saving_json(os.path.join(Files.output_dir,"info_dict_v2.json"),old)
    print(len(old))



if __name__ == "__main__":
    old = read_json(os.path.join(Files.work_dir, Files.old))
    new = read_json(os.path.join(Files.work_dir, Files.new))
    combine(old, new)
