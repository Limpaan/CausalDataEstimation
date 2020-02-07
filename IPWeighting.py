from Importer import Importer
from sklearn import linear_model, svm
import numpy as np
import matplotlib.pyplot as plt


def main():
    im = Importer()
    train = im.get_training_set()
    test = im.get_test_set()
    logis = linear_model.LogisticRegression()
    statevectormachine = svm.SVC(probability=True)

    for i in range(1):
        mu1 = train['mu1'][:, i]
        mu0 = train['mu0'][:, i]
        yf = train['yf'][:, i]
        ycf = train['ycf'][:, i]
        t = train['t'][:, i]
        x = train['x'][:, :, i]

        logis.fit(x, t)
        statevectormachine.fit(x, t)

        mu1_test = test['mu1'][:, i]
        mu0_test = test['mu0'][:, i]
        yf_test = test['yf'][:, i]
        ycf_test = test['ycf'][:, i]
        t_test = test['t'][:, i]
        x_test = test['x'][:, :, i]

        propensity_scores = logis.predict_proba(x_test)[:, 1]
        #propensity_scores = statevectormachine.predict_proba(x_test)[:, 1]

        n = len(t_test)
        weights = (1/n) * (t_test/propensity_scores + (1-t_test)/(1 - propensity_scores))
        yw = yf_test*weights
        #print(yw*t_test, yw*(1 - t_test))
        avg_yw1 = np.sum(yw*t_test)
        avg_yw0 = np.sum(yw*(1 - t_test))
        #print(weights)
        #print(avg_yw1, avg_yw0)
        print('Estimated ATE', avg_yw1 - avg_yw0)
        smu0 = sum(test['mu0'][:, i]) / len(test['mu0'][:, i])
        smu1 = sum(test['mu1'][:, i]) / len(test['mu1'][:, i])
        print("Real ATE: {}".format(str(smu1 - smu0)))


main()

