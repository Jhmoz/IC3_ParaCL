import os
import json
from tqdm import tqdm


class Files:
    work_dir = "../baike"
    old = "tag_summary_dict.json"
    new = "new_tag_summary.json"
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
        if lemma_id not in old:
            old[lemma_id] = new[lemma_id]
        else:
            # if name in old[lemma_id]["name"] or old[lemma_id]["name"] in name:
            if new[lemma_id]['tag'] != "None" and old[lemma_id]['tag']=='None':
                old[lemma_id]['tag'] = new[lemma_id]['tag']
    saving_json(os.path.join(Files.output_dir,"tag_summary_v2.json"),old)
    print(len(old))
    # print(unmatch_list[:20])
    # print(unmatch)



if __name__ == "__main__":
    old = read_json(os.path.join(Files.work_dir, Files.old))
    new = read_json(os.path.join(Files.work_dir, Files.new))
    combine(old, new)
