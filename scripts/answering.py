import pandas as pd
import numpy as np


def choose_from_proba(model, X_test):
    y_test = model.predict_proba(X_test)
    p_dislike = y_test[:, 0]
    p_like = y_test[:, 1]
    p_skip = y_test[:, 2]
    p_view = y_test[:, 3]

    exp = -10 * p_dislike - 0.1 * p_skip + 0.1 * p_view + 0.5 * p_like
    res = pd.Series(exp).apply(lambda x: 1 if x > 0 else -1)
    return res
