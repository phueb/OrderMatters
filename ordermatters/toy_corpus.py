from typing import List
from cached_property import cached_property
import random


class ToyCorpus:
    """
    create a collection of documents,
     each consisting of a string where artificial nouns are followed by a non-noun (other).
    example document: "n1 o5 n34 o82 n93 o3 n45 o11".
    the documents are sorted because:
     1. the population from which nouns are sampled is gradually increased
     2. the probability that only a select population of non-nouns can occur after a noun is gradually decreased

     these two constraints result in an ordered collection of documents,
     where the conditional entropy of nouns given the probability distribution over non-nouns decreases,
     while the joint entropy of nouns and non-nouns increases.
    """

    def __init__(self,
                 num_docs: int = 32,
                 doc_size: int = 100_000,
                 num_types: int = 4096,
                 num_nouns: int = 512,
                 divisor: int = 2,  # the larger, the more constrained are non-nouns
                 min_nouns: int = 100,
                 doc_offset: int = 6,  # the larger, the faster the noun population reaches its maximum
                 ) -> None:
        self.num_docs = num_docs
        self.doc_size = doc_size
        self.num_types = num_types
        self.num_nouns = num_nouns
        self.min_nouns = min_nouns
        self.doc_offset = doc_offset

        self.nouns = [f'n{i:0>6}' for i in range(self.num_nouns)]
        self.others = [f'o{i:0>6}' for i in range(self.num_types - self.num_nouns)]

        # a smaller set of non-nouns (non-nouns are preferentially sampled from this population in early documents)
        self.limited_others = [o for o in self.others if float(o[1:]) % divisor == 0]

        print('Initialized ToyCorpus with number of limited non-nouns:', len(self.limited_others))

    @cached_property
    def docs(self) -> List[str]:
        res = [self.doc(i) for i in range(self.num_docs)]
        return res

    def doc(self,
            doc_id,
            )-> str:

        assert 0 <= self.doc_offset <= self.num_docs

        # gradually increase noun population across consecutive documents
        limit = self.num_nouns * ((doc_id + self.doc_offset) / self.num_docs)
        nouns = self.nouns[:int(max(self.min_nouns, limit + 1))]

        # probability of constraining population of non-nouns
        prob = doc_id / self.num_docs

        # sample
        res = ''
        for n in range(self.doc_size):
            # sample noun randomly
            noun = random.choice(nouns)

            # sample next-word - sometimes from a limited population
            if random.random() > prob:
                others = self.limited_others
            else:
                others = self.others
            other = random.choice(others)
            res += f'{noun} {other} '  # whitespace after each
        return res