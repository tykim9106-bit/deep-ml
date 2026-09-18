import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    folds = np.array_split(indices, k)

    splits = []

    for i in range(k):
        test_indices = folds[i]

        train_indices = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        )

        splits.append(
            (train_indices.tolist(), test_indices.tolist())
        )

    return splits