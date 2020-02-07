from Importer import Importer
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

def main():
    im = Importer()
    train = im.get_training_set()
    test = im.get_test_set()
    logis = linear_model.LogisticRegression()

    for i in range(100):
        mu1 = train['mu1'][:, i]
        mu0 = train['mu0'][:, i]
        yf = train['yf'][:, i]
        ycf = train['ycf'][:, i]
        t = train['t'][:, i]
        x = train['x'][:, :, i]

        #x = x[:, 0][:, np.newaxis]
        logis.fit(x, t)

        mu1_test = test['mu1'][:, i]
        mu0_test = test['mu0'][:, i]
        yf_test = test['yf'][:, i]
        ycf_test = test['ycf'][:, i]
        t_test = test['t'][:, i]
        x_test = test['x'][:, :, i]

        mask = []
        for num in t_test:
            if num == 0:
                mask.append(False)
            else:
                mask.append(True)

        #x_test = x_test[:, 0][:, np.newaxis]

        ptx = logis.predict_proba(x_test)

        pt1 = np.sum(t_test)/len(t_test)
        pt0 = 1 - pt1
        ptx1 = ptx[mask]
        ptx0 = ptx[np.invert(mask)]

        #print('pt', pt1, pt0)

        #print('ptx', ptx1, ptx0)

        w1 = pt1/ptx1[:, 1]
        w0 = pt0/ptx0[:, 0]

        #print('w', w1, w0)

        yw1 = yf_test[mask]*w1
        yw0 = yf_test[np.invert(mask)]*w0
        #print('yf', yf_test)

        #print('yw', yw1, yw0)

        avg_yw1 = np.sum(yw1)/len(yw1)
        avg_yw0 = np.sum(yw0)/len(yw0)

        #print(avg_yw1, avg_yw0)
        print(avg_yw1 - avg_yw0)


main()

