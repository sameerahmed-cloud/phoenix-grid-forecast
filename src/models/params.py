"""The fixed LightGBM settings, in their own small file so the retraining job can use them without MLflow.

Fixed on purpose: tuning them against the test blocks would leak the answer.
"""

PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}