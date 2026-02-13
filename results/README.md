# Results Folder

This directory contains the generated outputs of the ensemble experiments.

## Contents

After running the project, this folder will include:

- **decision_boundaries.png**  
  Visualization of the decision regions produced by each ensemble model  
  (Bagging, Random Forest, AdaBoost, Gradient Boosting, and Stacking).

- **metrics.json**  
  A structured summary of evaluation metrics (accuracy scores).

- **metrics.csv**  
  The same metrics stored in tabular format for easier comparison.

## How to Generate

All results are automatically created when you run:

```bash
python -m src.main

