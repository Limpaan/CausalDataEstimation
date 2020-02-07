from Importer import Importer
from sklearn import linear_model
import numpy as np

im = Importer()
train = im.get_training_set()
test = im.get_test_set()

for i in range(1):
    mu1 = train['mu1'][:, i]
    mu0 = train['mu0'][:, i]
    yf = train['yf'][:, i]
    ycf = train['ycf'][:, i]
    t = train['t'][:, i]
    x = train['x'][:, :, i]

    mask = []
    for num in t:
        if num == 0:
            mask.append(False)
        else:
            mask.append(True)

    lm_t = linear_model.LinearRegression()
    model_t = lm_t.fit(x[mask], yf[mask])

    lm_nt = linear_model.LinearRegression()
    model_nt = lm_nt.fit(x[np.invert(mask)], yf[np.invert(mask)])

    te = sum(lm_t.predict(test['x'][:, :, i])) / len(test['x'][:, :, i])
    nte = sum(lm_nt.predict(test['x'][:, :, i])) / len(test['x'][:, :, i])
    print("Estimated ATE: {}".format(str(te - nte)))
    smu0 = sum(test['mu0'][:, i]) / len(test['mu0'][:, i])
    smu1 = sum(test['mu1'][:, i]) / len(test['mu1'][:, i])
    print("Real ATE: {}".format(str(smu1 - smu0)))