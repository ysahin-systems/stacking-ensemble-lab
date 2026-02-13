from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


def get_oof_meta_features(
    models,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
):
    

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    n = X.shape[0]          
    m = len(models)         

    
    X_meta = np.zeros((n, m), dtype=float)

   
    for model_idx, (name, model) in enumerate(models):

        
        oof_pred = np.zeros(n, dtype=float)

      
        for train_idx, val_idx in skf.split(X, y):
            
            mdl = clone(model)

           
            mdl.fit(X[train_idx], y[train_idx])

           
            proba = mdl.predict_proba(X[val_idx])[:, 1]

           
            oof_pred[val_idx] = proba

       
        X_meta[:, model_idx] = oof_pred

    return X_meta


def fit_base_and_make_meta(models, X_train, y_train, X_new):
    

    X_meta_new = np.zeros((X_new.shape[0], len(models)), dtype=float)
    fitted = []

    for i, (name, model) in enumerate(models):
        mdl = clone(model)
        mdl.fit(X_train, y_train)
        X_meta_new[:, i] = mdl.predict_proba(X_new)[:, 1]
        fitted.append((name, mdl))

    return X_meta_new, fitted
