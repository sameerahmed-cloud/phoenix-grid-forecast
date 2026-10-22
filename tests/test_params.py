import lightgbm as lgb

from src.models.params import PARAMS

# Valid LightGBM settings that the scikit-learn wrapper doesn't list by name.
EXTRA_VALID = {"verbose"}


def test_every_setting_is_a_real_lightgbm_parameter_name():
    # A misspelled name (for example one with a stray colon) is silently ignored and the default is used instead.
    known = set(lgb.LGBMRegressor().get_params()) | EXTRA_VALID
    assert set(PARAMS) <= known, f"not LightGBM parameters: {sorted(set(PARAMS) - known)}"


def test_the_settings_are_the_ones_the_models_were_trained_with():
    assert PARAMS["learning_rate"] == 0.05 and PARAMS["n_estimators"] == 400 and PARAMS["random_state"] == 42