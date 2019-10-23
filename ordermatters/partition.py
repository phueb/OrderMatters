

class Partitions:
    """
    collapse all documents (lines in a text file) to a single list of words.
    then, make equal sized partitions
    """

    def partition(self, token_ids):  # partitions should always contain token_ids rather than tokens
        result = []
        for i in range(0, len(token_ids), self.num_items_in_part):
            result.append(token_ids[i:i + self.num_items_in_part])
        return result

    def first_half(self):
        midpoint = len(self.train_terms.tokens) // 2
        result = self.train_terms.tokens[:midpoint]
        return result

    def second_half(self):
        midpoint = len(self.train_terms.tokens) // 2
        result = self.train_terms.tokens[-midpoint:]
        return result

    @property
    def ordered(self):
        raise NotImplementedError

    @property
    def reversed(self):
        raise NotImplementedError
