import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder, normalize

class gaussian_kde_classifier:

    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, x, y):
        assert len(y.shape) == 1
        assert len(x.shape) == 2
        self.nfeat = x.shape[1]
        self.kdes = []
        self.emps = []
        self.encoder.fit(y)
        self.n = self.encoder.classes_.shape[0]
        y1 = self.encoder.transform(y)
        net = float(y1.shape[0])
        for i in range(self.n):
            x1 = x[y1 == i]
            self.kdes.append(gaussian_kde(np.transpose(x1)))
            self.emps.append(float(x1.shape[0]) / net)
        self.emps = np.array(self.emps)

    def predict_probs(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.nfeat
        probs = []
        for i in range(self.n):
            probs.append(self.kdes[i].evaluate(np.transpose(x)) * self.emps[i])
        probs = np.transpose(np.array(probs))
        probs = probs / np.repeat(np.expand_dims(probs.sum(axis = 1), axis = 1), repeats = self.n, axis = 1)
        return probs

    def predict(self, x):
        return self.encoder.inverse_transform(np.argmax(self.predict_probs(x), axis = 1))

    def score(self, x, y):
        y1 = self.predict(x)
        y2 = self.encoder.transform(y)
        return float((y2 == y1).astype(int).sum()) / float(y.shape[0])
