# Script for MultiSpanQA evaluation
import re
import json
import string
import difflib
import warnings
import numpy as np
from typing import Dict, List, Set


def get_entities(label, token):
    label = ["O" if element == "-100" else element for element in label]
    prev_tag = 'O'
    begin_offset = 0
    chunks = []

    # check no ent
    if isinstance(label[0], list):
        for i, s in enumerate(label):
            if len(set(s)) == 1:
                chunks.append(('O', -i, -i))
    # for nested list
    if any(isinstance(s, list) for s in label):
        label = [item for sublist in label for item in sublist + ['O']]
    if any(isinstance(s, list) for s in token):
        token = [item for sublist in token for item in sublist + ['O']]

    for i, chunk in enumerate(label + ['O']):
        if chunk not in ["O", "B", "I"]:
            warnings.warn('{} seems not to be IOB tag.'.format(chunk))
        tag = chunk[0]
        if end_of_chunk(prev_tag, tag):
            chunks.append((' '.join(token[begin_offset:i]), begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag
    return chunks


def end_of_chunk(prev_tag: str, tag: str):
    """Determine if we are at the end of an answer chunk.

    :param prev_tag: previous tag
    :param tag: current tag
    """
    chunk_end = False
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True
    return chunk_end


def start_of_chunk(prev_tag: str, tag: str):
    """Determine if we are at the start of an answer chunk.

    :param prev_tag: previous tag
    :param tag: current tag
    """
    chunk_start = False
    if tag == 'B':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True
    return chunk_start


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.

    :param s: input string
    """

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_scores(golds: Dict[str, Set], preds: Dict[str, Set], eval_type: str = 'em', average: str = 'micro'):
    """Compute precision, recall and exact match (or f1) metrics.

    :param golds: dictionary of gold XX
    :param preds: dictionary of predictions
    :param eval_type: Evaluation type. Can be either "em" or "overlap".
    """
    nb_gold = 0
    nb_pred = 0
    nb_correct = 0
    nb_correct_p = 0
    nb_correct_r = 0
    for keys in list(golds.keys()):
        gold = golds[keys]
        pred = preds[keys]
        nb_gold += max(len(gold), 1)
        nb_pred += max(len(pred), 1)
        if eval_type == 'em':
            if len(gold) == 0 and len(pred) == 0:
                # Exact match no answer case
                nb_correct += 1
            else:
                # Exact match comparison between two sets
                nb_correct += len(gold.intersection(pred))
        else:
            p_score, r_score = count_overlap(gold, pred)
            nb_correct_p += p_score
            nb_correct_r += r_score

    if eval_type == 'em':
        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_gold if nb_gold > 0 else 0
    else:
        p = nb_correct_p / nb_pred if nb_pred > 0 else 0
        r = nb_correct_r / nb_gold if nb_gold > 0 else 0

    f = 2 * p * r / (p + r) if p + r > 0 else 0

    return p, r, f


def count_overlap(gold: set, pred: set):
    """Count the overlap of the gold answer and the predicted answer.

    :param gold: Set of gold answers
    :param pred: Set of predicted answers
    """
    # Correct no answer prediction
    if len(gold) == 0 and (len(pred) == 0 or pred == {""}):
        return 1, 1

    # Incorrect no answer prediction
    elif len(gold) == 0 or (len(pred) == 0 or pred == {""}):
        return 0, 0

    # NOTE: Since it is possible to return multiple spans it is not clear which spans from pred should be compared to
    #       each span in gold. So all are compared and the highest precision and recall are taken.
    p_scores = np.zeros((len(gold), len(pred)))
    r_scores = np.zeros((len(gold), len(pred)))
    for i, gold_str in enumerate(gold):
        for j, pred_str in enumerate(pred):
            seq_matcher = difflib.SequenceMatcher(None, gold_str, pred_str)
            _, _, longest_len = seq_matcher.find_longest_match(0, len(gold_str), 0, len(pred_str))
            p_scores[i][j] = longest_len / len(pred_str) if longest_len > 0 else 0
            r_scores[i][j] = longest_len / len(gold_str) if longest_len > 0 else 0

    p_score = sum(np.max(p_scores, axis=0))
    r_score = sum(np.max(r_scores, axis=1))

    return p_score, r_score


def read_gold(gold_file):
    """Read the gold file  and Classify examples according to the type of qa

    :param gold_file: file path to the file with the golden answers
    """
    types = ["d","e","n"]
    with open(gold_file) as f:
        data = json.load(f)
        golds = {"all":{}}
        for example_id in data:
            piece = data[example_id]
            # valid_label=["O" if element=="-100" else element for element in piece["label"]]
            if piece["type"] not in golds:
                golds[piece["type"]] = {}
            prediction_entity_format = piece["gold"]
            golds[piece["type"]][example_id] = prediction_entity_format
            golds["all"][example_id] = prediction_entity_format
    return golds


def read_pred(pred_file):
    """Read the prediction file

    :param pred_file: file path to the prediction file
    """
    with open(pred_file) as f:
        preds = json.load(f)
    return preds

def get_match_pred(gold_grouped, pred):
    pred_grouped = {}
    for example_type in gold_grouped:
        if example_type not in pred_grouped:
            pred_grouped[example_type] = {}
        for example_id in gold_grouped[example_type]:
            gold_example=gold_grouped[example_type][example_id]
            pred_grouped[example_type][example_id] = pred[example_id]
        print(example_type,len(pred_grouped[example_type]))
    return pred_grouped

def reformat_eval_result(results_grouped):
    res={}
    for qa_type in results_grouped:
        for metric in results_grouped[qa_type]:
            key=f"{qa_type}_{metric}"
            res[key]=results_grouped[qa_type][metric]
    return res

def multi_span_evaluate(preds: Dict[str, Set], golds: Dict[str, Set]):
    assert len(preds) == len(golds)
    assert preds.keys() == golds.keys()

    # Normalize the answer
    for k, v in golds.items():
        golds[k] = set(map(lambda x: normalize_answer(x), v))
    for k, v in preds.items():
        preds[k] = set(map(lambda x: normalize_answer(x), v))

    # Evaluate
    em_p, em_r, em_f = compute_scores(golds, preds, eval_type='em')  # type: ignore
    overlap_p, overlap_r, overlap_f = compute_scores(golds, preds, eval_type='overlap')  # type: ignore
    result = {'exact_match_precision': 100 * em_p,
              'exact_match_recall': 100 * em_r,
              'exact_match_f1': 100 * em_f,
              'overlap_precision': 100 * overlap_p,
              'overlap_recall': 100 * overlap_r,
              'overlap_f1': 100 * overlap_f}
    return result

def multi_span_evaluate_with_types(preds, golds):
    results_grouped = {}
    for qa_type in golds:
        result = multi_span_evaluate(preds[qa_type], golds[qa_type])
        results_grouped[qa_type] = result
    results=reformat_eval_result(results_grouped)
    return results


def multi_span_evaluate_from_file(pred_file, gold_file,output_file):
    preds = read_pred(pred_file)
    golds_grouped = read_gold(gold_file)
    preds_grouped = get_match_pred(golds_grouped, preds)
    results_grouped = {}
    for qa_type in golds_grouped:
        result = multi_span_evaluate(preds_grouped[qa_type], golds_grouped[qa_type])
        results_grouped[qa_type] = result
    results=reformat_eval_result(results_grouped)
    with open(output_file,"w") as f:
        json.dump(results,f)
    return results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', default="", type=str)
    parser.add_argument('--gold_file', default="", type=str)
    parser.add_argument('--output_file' ,type=str)
    args = parser.parse_args()
    eval_result = multi_span_evaluate_from_file(args.pred_file, args.gold_file, args.output_file)
    for k, v in eval_result.items():
        print(f"{k}: {v}")
