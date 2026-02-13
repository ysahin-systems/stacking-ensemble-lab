from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary_from_predict(predict_fn, X, y, title: str, ax=None):
   

   
    if ax is None:
        ax = plt.gca()

    
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

   
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    
    grid = np.c_[xx.ravel(), yy.ravel()]  

   
    Z = predict_fn(grid)                  
    Z = Z.reshape(xx.shape)
   
    ax.contourf(xx, yy, Z, alpha=0.30)

    
    ax.scatter(X[:, 0], X[:, 1], c=y, s=12)

    
    ax.set_title(title)


def save_fig(path: str, dpi: int = 180):
    
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
