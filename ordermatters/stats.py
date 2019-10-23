from sklearn.feature_extraction.text import CountVectorizer


def calc_num_unique_ngrams_in_part(self, ngram_range, part):
    v = CountVectorizer(ngram_range=ngram_range)
    a = v.build_analyzer()
    tokens = [self.train_terms.types[term_id] for term_id in part]
    ngrams = a(' '.join(tokens))  # requires string
    result = len(set(ngrams))
    return result