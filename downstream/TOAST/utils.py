import difflib
import json
import string
import warnings
from itertools import chain
import pickle
import re
from typing import Dict, Set

import numpy as np


def is_nested_list(lst):
    if isinstance(lst, list):
        if len(lst) > 0:
            for elem in lst:
                if isinstance(elem, list):
                    return True
    return False


def subtract_offset_for_nested_list_elements(span, offset):
    if span is None:
        return [-1, -1]
    if not is_nested_list(span):
        if isinstance(span, list):
            return [elem - offset for elem in span]
        else:
            return span - offset
    else:
        return [subtract_offset_for_nested_list_elements(elem, offset) for elem in span]


def valid_nested_list_range(span, lower_bound=None, upper_bound=None):
    if not is_nested_list(span):
        if isinstance(span, list):
            for elem in span:
                if lower_bound is not None and elem < lower_bound:
                    return False
                if upper_bound is not None and elem > upper_bound:
                    return False
            return True
        else:
            if lower_bound is not None and span < lower_bound:
                return False
            if upper_bound is not None and span > upper_bound:
                return False
            return True
    else:
        for elem in span:
            if valid_nested_list_range(elem, lower_bound, upper_bound) is False:
                return False
        return True


def text_span_flatten(text_span, results):
    if text_span is None:
        return None

    if is_nested_list(text_span):
        for elem in text_span:
            text_span_flatten(elem, results)
    else:
        results.extend(text_span[:])


def leveled_text_span_flatten(text_span, level, pos, results):
    if text_span is None:
        return None

    if is_nested_list(text_span):
        for pos, elem in enumerate(text_span):
            leveled_text_span_flatten(elem, level + 1, pos, results)
    else:
        results.extend([elem + f"<level{level}><pos{pos}>" for elem in text_span])


def get_first_answer_span(answer_spans):
    if answer_spans is None:
        return [-1, -1]
    if not is_nested_list(answer_spans):
        return answer_spans
    else:
        return get_first_answer_span(answer_spans[0])


def get_flatten_answer_spans(answer_spans):
    if answer_spans is None:
        return [-1, -1]
    if not is_nested_list(answer_spans):
        return answer_spans
    else:
        res = []
        for elem in answer_spans:
            if is_nested_list(elem):
                res += get_flatten_answer_spans(elem)
            elif isinstance(elem, list):
                res += elem
            else:
                res += [elem]
        return res


def get_first_answer_span_text(answer_span_texts):
    if answer_span_texts is None:
        return None
    if isinstance(answer_span_texts, str):
        return answer_span_texts
    else:  # non-empty list
        return get_first_answer_span_text(answer_span_texts[0])


def get_flatten_answer_span_texts(answer_span_texts):
    if answer_span_texts is None:
        return None
    if not is_nested_list(answer_span_texts):
        return answer_span_texts
    else:
        res = []
        for elem in answer_span_texts:
            if is_nested_list(elem):
                res += get_flatten_answer_spans(elem)
            elif isinstance(elem, list):
                res += elem
            else:
                res += [elem]
        return res


def nested_list_shape_match_or_not(a, b):
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        elif len(a) != 0:
            return nested_list_shape_match_or_not(a[0], b[0]) and nested_list_shape_match_or_not(a[1:], b[1:])
    else:
        return True


def match_answer_span_and_text(answer_span, answer_text):
    if answer_text is None and (answer_span is None or answer_span == [-1, -1]):
        return True
    l1 = len(get_flatten_answer_spans(answer_span)) // 2
    l2 = len(get_flatten_answer_span_texts(answer_text))
    return l1 == l2


def built_filtered_data(input_file, output_file):
    orig_sample = pickle.load(open(input_file, "rb"))
    invalids = [i for i, example in enumerate(orig_sample) if
                not match_answer_span_and_text(example['answer_span'], example['answer_text'])]
    filtered_examples = [example for i, example in enumerate(orig_sample) if i not in invalids]
    pickle.dump(filtered_examples, open(output_file, 'wb'))


def get_entities(label, token):
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
    #breakpoint()
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
            p_scores[i][j] = longest_len/len(pred_str) if longest_len > 0 else 0
            r_scores[i][j] = longest_len/len(gold_str) if longest_len > 0 else 0

    p_score = sum(np.max(p_scores, axis=0))
    r_score = sum(np.max(r_scores, axis=1))

    return p_score, r_score


def read_gold(gold_file):
    """Read the gold file

    :param gold_file: file path to the file with the golden answers
    """
    with open(gold_file) as f:
        data = json.load(f)['data']
        golds = {}
        for piece in data:
            golds[piece['example_id']] = set(map(lambda x: x[0], get_entities(piece['label'], piece['context'])))
    return golds


def read_pred(pred_file):
    """Read the prediction file

    :param pred_file: file path to the prediction file
    """
    with open(pred_file) as f:
        preds = json.load(f)
    return preds


def multi_span_evaluate_from_file(pred_file, gold_file):
    preds = read_pred(pred_file)
    golds = read_gold(gold_file)
    result = multi_span_evaluate(preds, golds)
    return result


def multi_span_evaluate(preds: Dict[str, Set], golds: Dict[str, Set]):
    assert len(preds) == len(golds)
    # print(preds.keys())
    # print(golds.keys())
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
