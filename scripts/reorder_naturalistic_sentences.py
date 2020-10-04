from typing import List
from random import shuffle

from preppy.docs import split_into_sentences

from ordermatters import configs

# CORPUS_NAME = 'newsela'
CORPUS_NAME = 'childes-20191112'
WORDS_NAME = 'sem-4096'
RIGHT_NEIGHBORS = ['.', '?', 'with', 'OOV', 'right']  # ['.', '?', 'with', 'that']
REMOVE_NUMBER_WORDS = True


def make_lines_in_part(sentences_in_part: List[List[str]],
                       num_docs_in_part: int,
                       num_sentences_in_doc: int,
                       ) -> List[str]:
    res = []
    for doc_id in range(num_docs_in_part):
        _line: str = ''
        for _ in range(num_sentences_in_doc):
            _s: List[str] = sentences_in_part.pop()
            _s_string: str = ' '.join(_s)
            _line += f'{_s_string} '
        _line += '\n'
        res.append(_line)

    # put leftover sentences in last doc
    _line: str = ''
    for _ in range(len(sentences_in_part)):
        _s: List[str] = sentences_in_part.pop()
        _s_string: str = ' '.join(_s)
        _line += f'{_s_string} '
    _line += '\n'
    res.append(_line)

    assert len(sentences_in_part) == 0

    return res


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
num_docs_in_part1 = num_docs // 2
num_docs_in_part2 = num_docs // 2
num_sentences1_per_doc = len(part1_sentences) // num_docs_in_part1
num_sentences2_per_doc = len(part2_sentences) // num_docs_in_part2
lines1 = make_lines_in_part(part1_sentences, num_docs_in_part1, num_sentences1_per_doc)
lines2 = make_lines_in_part(part2_sentences, num_docs_in_part2, num_sentences2_per_doc)


# save to file
out_path = configs.Dirs.root / 'reordered_corpora' / f'{CORPUS_NAME}-ce.txt'
out_file = out_path.open('w')
for n, line in enumerate(lines1 + lines2):
    # print(f'Writing line {n}')
    out_file.write(line)
out_file.close()