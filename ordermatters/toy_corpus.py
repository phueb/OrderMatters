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

     these two constraints rsult in an ordered collection of documents,
     where the conditional entropy of nouns given the probability distribution over non-nouns decreases,
     while the joint entropy of nouns and non-nouns increases.
    """

    def __init__(self,
                 num_parts: int = 32,
                 part_size: int = 100_000,
                 num_types: int = 4096,
                 num_nouns: int = 512,
                 ) -> None:
        self.num_parts = num_parts
        self.part_size = part_size
        self.num_types = num_types
        self.num_nouns = num_nouns

        self.nouns = [f'n{i}' for i in range(self.num_nouns)]
        self.others = [f'o{i}' for i in range(self.num_types - self.num_nouns)]

        self.limited_others = [o for o in self.others if float(o[1:]) % 2 == 0]

    @cached_property
    def docs(self) -> List[str]:
        res = [self.doc(i) for i in range(self.num_parts)]
        return res

    def doc(self,
            part_id,
            min_nouns: int = 100,
            part_offset: int = 6,  # the larger, the faster the noun population reaches its maximum
            )-> str:

        assert 0 < part_offset < self.num_parts

        # gradually increase noun population across consecutive documents
        limit = self.num_nouns * ((part_id + part_offset) / self.num_parts)
        nouns = self.nouns[:int(max(min_nouns, limit))]

        # TODO manipulate non-noun population - make it conditional on nouns
        prob = part_id / self.num_parts
        print(prob)

        res = ''
        for n in range(self.part_size):
            # sample noun randomly
            noun = random.choice(nouns)

            # sample next-word - sometimes from a limited population
            if random.random() > prob:  # make non-noun population conditional on nouns
                others = self.limited_others
                print('Making conditional', len(others))
            else:
                others = self.others
                print('Leaving as is', len(others))

            other = random.choice(others)
            res += f'{noun} {other} '  # whitespace after each
        return res