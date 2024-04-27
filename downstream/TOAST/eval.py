import re
import string
import warnings


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
        exclude = set(string.punctuation+"﹒﹔﹖﹗．；。！？"+'"’”」』'+'"‘“「『')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


if __name__ == "__main__":
    print(get_entities(['O', 'O', 'B', 'I', 'O', 'B', 'B', 'O'], ['It', 'is', 'Carl', 'Marks', ',', 'U.S.A', 'president', '.']))