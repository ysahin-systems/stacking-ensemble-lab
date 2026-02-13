from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)


def build_base_models(seed: int = 42):
  
    tree = DecisionTreeClassifier(
        max_depth=5,       
        random_state=seed
    )


    svm = Pipeline([
        ("scaler", StandardScaler()), 
        ("svc", SVC(
            C=3.0,
            gamma="scale",
            probability=True,
            random_state=seed
        ))
    ])


    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=3000,
            random_state=seed
        ))
    ])

    
    return [
        ("tree", tree),
        ("svm", svm),
        ("lr", lr),
    ]


def build_meta_model(seed: int = 42):

    return LogisticRegression(max_iter=3000, random_state=seed)


def build_baselines(seed: int = 42):

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=seed),
        n_estimators=200,
        bootstrap=True,
        random_state=seed,
        n_jobs=-1
    )


    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        oob_score=True
    )


    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
        n_estimators=300,
        learning_rate=0.5,
        random_state=seed
    )

    
    gbdt = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=2,
        random_state=seed
    )

    return {
        "bagging": bagging,
        "random_forest": rf,
        "adaboost": ada,
        "gbdt": gbdt,
    }
