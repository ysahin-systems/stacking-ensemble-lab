from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from .data import make_dataset
from .models import build_base_models, build_meta_model, build_baselines
from .stacking_oof import get_oof_meta_features, fit_base_and_make_meta
from .plotting import plot_decision_boundary_from_predict, save_fig
from .utils import ensure_dir, save_json, save_csv


def main():

    seed = 42
    X, y, X_train, X_test, y_train, y_test = make_dataset(
        n_samples=2000,
        noise=0.30,
        seed=seed,
        test_size=0.25
    )


    base_models = build_base_models(seed=seed)
    meta_model = build_meta_model(seed=seed)


    X_meta_train = get_oof_meta_features(
        base_models, X_train, y_train,
        n_splits=5,
        seed=seed
    )


    meta_model.fit(X_meta_train, y_train)


    X_meta_test, fitted_base = fit_base_and_make_meta(
        base_models, X_train, y_train, X_test
    )

    y_pred_stack = meta_model.predict(X_meta_test)
    acc_stack = accuracy_score(y_test, y_pred_stack)

 
    baselines = build_baselines(seed=seed)
    baseline_scores = {}

    for name, model in baselines.items():
        model.fit(X_train, y_train)
        baseline_scores[name] = model.score(X_test, y_test)

   
    rf_oob = None
    try:
        rf_oob = baselines["random_forest"].oob_score_
    except Exception:
        rf_oob = None


    out_dir = ensure_dir("results")

    metrics = {
        "stacking_oof_accuracy": float(acc_stack),
        "baseline_test_scores": {k: float(v) for k, v in baseline_scores.items()},
        "random_forest_oob_score": None if rf_oob is None else float(rf_oob),
    }

    save_json(metrics, out_dir / "metrics.json")

    rows = [{"model": "stacking_oof", "test_accuracy": float(acc_stack)}]
    for k, v in baseline_scores.items():
        rows.append({"model": k, "test_accuracy": float(v)})
    save_csv(rows, out_dir / "metrics.csv")

    print("Stacking OOF Accuracy:", acc_stack)
    print("Baselines:", baseline_scores)
    if rf_oob is not None:
        print("RandomForest OOB Score:", rf_oob)

  
    def stacking_predict(grid_2d: np.ndarray) -> np.ndarray:
        
        X_meta_grid, _ = fit_base_and_make_meta(base_models, X_train, y_train, grid_2d)
        return meta_model.predict(X_meta_grid)

    plt.figure(figsize=(14, 8))

    
    ax = plt.subplot(2, 3, 1)
    plot_decision_boundary_from_predict(
        lambda g: baselines["bagging"].predict(g),
        X_test, y_test, "Bagging (Test set)", ax=ax
    )

    
    ax = plt.subplot(2, 3, 2)
    plot_decision_boundary_from_predict(
        lambda g: baselines["random_forest"].predict(g),
        X_test, y_test, "Random Forest (Test set)", ax=ax
    )

    
    ax = plt.subplot(2, 3, 3)
    plot_decision_boundary_from_predict(
        lambda g: baselines["adaboost"].predict(g),
        X_test, y_test, "AdaBoost (Test set)", ax=ax
    )

    
    ax = plt.subplot(2, 3, 4)
    plot_decision_boundary_from_predict(
        lambda g: baselines["gbdt"].predict(g),
        X_test, y_test, "Gradient Boosting (Test set)", ax=ax
    )

    
    ax = plt.subplot(2, 3, 5)
    plot_decision_boundary_from_predict(
        stacking_predict,
        X_test, y_test, "Stacking OOF (Test set)", ax=ax
    )

    save_fig(str(out_dir / "decision_boundaries.png"))
    plt.close()


if __name__ == "__main__":
    main()
