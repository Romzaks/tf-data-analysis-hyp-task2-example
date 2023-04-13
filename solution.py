import pandas as pd
import numpy as np
from hyppo.ksample import Energy, MMD
from statsmodels.distributions.empirical_distribution import ECDF

chat_id = 333357078 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool:
    alpha = 0.08
    pl = lambda x, y: MMD(compute_kernel="rbf", gamma=1/10).test(x, y)[1]
    p = pl(x, y)
    if p <= alpha:
        return True
    else:
        return False
