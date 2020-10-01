
import numpy as np
from pyitlib import discrete_random_variable as drv


def compute_joint_entropy(test_word_windows: np.ndarray,
                          col_id: int,
                          ) -> float:
    """
    compute conditional entropy of test_word words, given distribution of words that follow them.
    
    """

    # joint entropy
    x = test_word_windows[:, -2]  # test word
    y = test_word_windows[:, col_id]  # neighbor
    x_y = np.vstack((x, y))
    jei = drv.entropy_joint(x_y).item()

    return jei


def compute_y_entropy(test_word_windows: np.ndarray,
                      col_id: int,
                      ) -> float:
    """
    compute  entropy of next-words
    """

    y = test_word_windows[:, col_id]  # neighbor
    yei = drv.entropy_joint(y).item()

    return yei


def compute_unconditional_entropy(windows: np.ndarray,
                                  col_id: int,
                                  ) -> float:
    """
    compute entropy of all words in a part
    """
    y = windows[:, col_id]
    uei = drv.entropy_joint(y).item()

    return uei


def compute_information_interaction(test_word_windows: np.ndarray,
                                    ) -> float:
    """
    compute info-interaction between test_word, left-word, and right-word probability distributions
    """

    assert test_word_windows.shape[1] >= 3

    # WARNING: for this to work, windows must have dim1 >=3
    x = test_word_windows.T  # 3 rows, where each row contains realisations of a distinct random variable
    iii = drv.information_interaction(x)

    return iii


def compute_mutual_information_difference(windows: np.ndarray,
                                          test_word_windows: np.ndarray,
                                          col_id: int,
                                          ) -> float:
    """
    compute subtraction: I(all X;Y) - I(test word X;Y)
    """

    # val1
    x1 = windows[:, -2]  # all words
    y1 = windows[:, col_id]  # neighbors
    val1i = drv.information_mutual_normalised(x1, y1).item()

    # val2
    x2 = test_word_windows[:, -2]  # test_words
    y2 = test_word_windows[:, col_id]  # neighbors
    val2i = drv.information_mutual_normalised(x2, y2).item()

    # windows with
    mdi = val1i - val2i

    return mdi
