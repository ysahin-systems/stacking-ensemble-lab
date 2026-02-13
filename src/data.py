from __future__ import annotations

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def make_dataset(
    n_samples: int = 2000,
    noise: float = 0.30,
    seed: int = 42,
    test_size: float = 0.25,
):
   

  
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=seed
    )


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )


    return X, y, X_train, X_test, y_train, y_test
