from typing import List
from cached_property import cached_property
import random


class ToyCorpus:
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
        self.non_nouns = [f'w{i}' for i in range(self.num_types - self.num_nouns)]

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

        res = ''
        for n in range(self.part_size):
            noun = random.choice(nouns)
            non_noun = random.choice(self.non_nouns)
            res += f'{noun} {non_noun} '  # whitespace after each
        return res