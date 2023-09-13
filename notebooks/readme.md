>This folder groops the jupyter notebooks used for EDA, Hyperparameter search and XGBoost regression, and a Pytorch LSTM model for multivariate time series gap filling.

⚠️ ***np.int*** was a deprecated alias for the builtin ***int*** but still used in the current version of **skopt**. To avoid an error when performing the optimization in the ***2_xgb_bayes_hp_tuning.ipynb*** notebook, replace *np.int* with *int* by itself or `np.int64` in `skopt/space/transforms.py`.
