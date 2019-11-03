from typing import List


def reorder_docs_from_midpoint(docs: List[str]
                               ) -> List[str]:
    """
    reorder docs such that first docs are docs that are most central
    """
    # start, middle, end
    s = 0
    e = len(docs)
    m = e // 2

    res = []
    for i, j in zip(range(m, e + 1)[::+1],
                    range(s, m + 0)[::-1]):
        res += [docs[i], docs[j]]

    assert len(res) == len(docs)

    return res


def reorder_docs_from_ends(docs: List[str]
                           ) -> List[str]:
    """
    reorder docs such that first docs are docs that are from ends
    """
    # start, middle, end
    s = 0
    e = len(docs)
    m = e // 2

    res = []
    for i, j in zip(range(m, e + 0)[::-1],
                    range(s, m + 1)[::+1]):
        res += [docs[i], docs[j]]

    assert len(res) == len(docs)

    return res
