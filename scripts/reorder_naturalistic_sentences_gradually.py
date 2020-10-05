from typing import List
import random
import numpy as np
from random import shuffle

from preppy.docs import split_into_sentences

from ordermatters import configs

# CORPUS_NAME = 'newsela'
CORPUS_NAME = 'childes-20191112'
WORDS_NAME = 'sem-4096'
RIGHT_NEIGHBORS = ['.', '?', 'with', 'OOV', 'right']  # ['.', '?', 'with', 'that']
REMOVE_NUMBER_WORDS = True


# load all sentences
corpus_path = configs.Dirs.corpora / f'{CORPUS_NAME}.txt'
text_in_file = corpus_path.read_text()
tokens = text_in_file.replace('\n', ' ').split()
sentences = split_into_sentences(tokens, punctuation={'.', '!', '?'})
print(f'Found {len(sentences)} sentences')
shuffle(sentences)

# load test words
test_words_file_path = configs.Dirs.words / f'{CORPUS_NAME}-{WORDS_NAME}.txt'
test_words = set([w for w in test_words_file_path.read_text().split('\n')])
print('num test_words', len(test_words))
if REMOVE_NUMBER_WORDS:  # number words are not nouns
    number_words_file_path = configs.Dirs.words / f'{CORPUS_NAME}-numbers.txt'
    for number in [w for w in number_words_file_path.read_text().split('\n')]:
        if number in test_words:
            test_words.remove(number)
            print('Removed', number)

# sort sentences by neighbor
si = iter(sentences)
rns = set(RIGHT_NEIGHBORS)
rn1_sentences = []
rn2_sentences = []
other_sentences = []
while True:
    try:
        s = next(si)
    except StopIteration:
        break

    for loc, w in enumerate(s):
        if w in test_words:
            if s[loc + 1] in rns:
                rn1_sentences.append(s)
            else:
                rn2_sentences.append(s)
            break
    else:
        other_sentences.append(s)

print(f'num sentences in part1={len(rn1_sentences):>9,}')
print(f'num sentences in part2={len(rn2_sentences):>9,}')
print(f'num sentences in both ={len(other_sentences):>9,}')

# combine with other_sentences
half = len(other_sentences) // 2
part1_sentences = other_sentences[:+half] + rn1_sentences
part2_sentences = other_sentences[-half:] + rn2_sentences
shuffle(part1_sentences)
shuffle(part2_sentences)

# make lines
print('Making lines...')
num_docs = len(text_in_file.split('\n'))
num_total_sentences = len(part1_sentences) + len(part2_sentences)
num_sentences_per_doc = num_total_sentences // num_docs
lines = []


def make_line(sentences1: List[List[str]],
              sentences2: List[List[str]],
              prob1: float,  # probability of selecting a sentence from sentences1
              num_sentences: int,
              ):
    res: str = ''
    for _ in range(num_sentences):
        if random.random() < prob1:
            sentence = ' '.join(sentences1.pop())
        else:
            sentence = ' '.join(sentences2.pop())
        res += sentence + ' '
    res += '\n'

    return res


for doc_id, prob1 in enumerate(np.linspace(1, 0, num_docs)):
    try:
        line = make_line(part1_sentences, part2_sentences, prob1, num_sentences_per_doc)
    except IndexError:  # pop from empty list
        break
    lines.append(line)

# put leftover sentences in last doc/line
if len(part1_sentences) > 0:
    last_line = make_line(part1_sentences, [], 1.0, len(part1_sentences))
elif len(part2_sentences) > 0:
    last_line = make_line(part2_sentences, [], 1.0, len(part2_sentences))
else:
    raise RuntimeError
lines.append(last_line)
assert len(part1_sentences) == 0
assert len(part2_sentences) == 0

# save to file
out_path = configs.Dirs.root / 'reordered_corpora' / f'{CORPUS_NAME}-ce-g.txt'
out_file = out_path.open('w')
for n, line in enumerate(lines):
    # print(f'Writing line {n}')
    out_file.write(line)
out_file.close()