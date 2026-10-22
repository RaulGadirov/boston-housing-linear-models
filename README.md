# Boston Housing Price Prediction: Linear Model Comparison

Predicting median house value (`medv`) on the Boston Housing dataset with linear models, custom feature engineering and cross-validation. Compares **Linear, Ridge, Lasso, ElasticNet and Polynomial (Ridge)** regression.

## Highlights

- Missing `rm` values (5 rows) filled with `IterativeImputer`, fitted on features only (target excluded to avoid leakage)
- Target transformed with `log1p`, predictions converted back with `expm1` and evaluated in the **original price scale**
- ~40 engineered features: ratios (`tax_per_rm`, `nox_per_dis`), interactions (`rm_x_lstat`, `nox_x_indus`), log transforms, and binary flags for dataset quirks (`rad == 24`, `ptratio == 20.2`, `age == 100`, `tax == 666`)
- 10-fold cross-validation with a custom scorer (R² on the original scale)
- Regularization strength tuned via `RidgeCV`, `LassoCV`, `ElasticNetCV`
- Polynomial (degree 2) features with `StandardScaler` before and after expansion + `RidgeCV`

## Results

Hold-out test set (25%), original price scale:

| Model      | MAE    | RMSE   | R² (%) |
|------------|--------|--------|--------|
| Linear     | 2.177  | 3.418  | **83.32** |
| Lasso      | 2.206  | 3.527  | 82.24  |
| Ridge      | 2.197  | 3.534  | 82.17  |
| Polynomial | **2.039** | 3.540 | 82.10 |
| ElasticNet | 2.204  | 3.558  | 81.92  |

Cross-validation R² on the training set: Polynomial 0.860, Linear 0.795, Ridge 0.794, ElasticNet 0.791, Lasso 0.790.

**Takeaways**

- Plain Linear regression has the best test R² and RMSE; Polynomial has the lowest MAE.
- Polynomial CV score (0.86) is noticeably higher than its test score (0.82): a sign of mild overfitting on a small dataset (506 rows).
- Unregularized polynomial regression blew up (R² around -10^5) because `expm1` amplifies extreme predictions on outliers; adding Ridge regularization and rescaling fixed it.
- Coefficients of correlated engineered features (`nox`, `nox_per_dis`, ...) cancel each other, so single weights should not be read as feature importance. Permutation importance is a better tool here.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter

## Project structure

```
├── Linear_finalplease_model_comparisonFinal.ipynb   # EDA, feature engineering, models, evaluation
├── BostonHousing.csv                                 # dataset
└── README.md
```

## How to run

```bash
git clone https://github.com/RaulGadirov/boston-housing-linear-models.git
cd boston-housing-linear-models
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Update the CSV path in the notebook (`/content/BostonHousing.csv` is a Google Colab path) to your local file.

## Possible improvements

- Fit feature engineering inside a `Pipeline` so test preprocessing is guaranteed identical to training
- Compare with tree-based models (Random Forest, Gradient Boosting)
- Permutation importance / SHAP for reliable feature attribution
- Prune correlated engineered features (VIF) and tune polynomial degree

## Author

**Raul Gadirov**: [GitHub](https://github.com/RaulGadirov) · [Kaggle](https://www.kaggle.com/raulgv)
