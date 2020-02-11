from Importer import Importer
from sklearn import linear_model
import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

def main():
    im = Importer()
    train = im.get_training_set()
    test = im.get_test_set()

    for i in range(100):
        t = train['t'][:, i]
        x = train['x'][:, :, i]
        yf = train['yf'][:, i]

        x_test = test['x'][:, :, i]
        t_test = test['t'][:, i]
        yf_test = test['yf'][:, i]

        propensityscore_model = linear_model.LogisticRegression(solver='lbfgs')
        propensityscore_model.fit(x, t)

        propensityscore = propensityscore_model.predict_proba(x_test)[:, 1]
        propensityscore = np.maximum(0.05, propensityscore)
        propensityscore = np.minimum(0.95, propensityscore)
        ips_treated = sum(t_test / propensityscore)
        ips_control = sum((1 - t_test) / (1 - propensityscore))
        n = len(t_test)
        n_treated = sum(t_test)
        n_control = n - n_treated

        ips_weight = (1 / n) * (t_test / propensityscore + (1 - t_test) / (1 - propensityscore))

        p_treatment = sum(t_test) / n

        ips_stabilized_weight = (1 / n) * (t_test / propensityscore * p_treatment + (1 - t_test) / (1 - propensityscore) * (1 - p_treatment))

        d_y = ips_weight * t_test * yf_test
        d_bar_y = ips_weight * (1 - t_test) * yf_test

        ate = sum(d_y) - sum(d_bar_y)
        print(str(ate) + "vs" + str(sum(test['mu1'][:, i]) / len(test['mu1'][:, i]) - sum(test['mu0'][:, i]) / len(test['mu0'][:, i])))

main()

