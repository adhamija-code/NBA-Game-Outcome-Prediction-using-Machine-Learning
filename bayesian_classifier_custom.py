import numpy as np
from scipy.stats import multivariate_normal, norm
from sklearn.neighbors import KernelDensity

class BayesianClassifier:
    def __init__(self, mode='multivariate'):
        self.mode = mode
        self.classes = None
        self.class_priors = {}
        self.class_params = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / len(X)
            if self.mode == 'multivariate':
                mean = np.mean(X_cls, axis=0)
                cov = np.cov(X_cls, rowvar=False) + 1e-6 * np.eye(X.shape[1])
                self.class_params[cls] = (mean, cov)
            elif self.mode == 'naive':
                means = np.mean(X_cls, axis=0)
                stds = np.std(X_cls, axis=0) + 1e-6
                self.class_params[cls] = (means, stds)
            elif self.mode == 'nonparametric':
                kde = KernelDensity(kernel='gaussian', bandwidth=1.0)
                kde.fit(X_cls)
                self.class_params[cls] = kde

    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            prior = self.class_priors[cls]
            if self.mode == 'multivariate':
                mean, cov = self.class_params[cls]
                likelihood = multivariate_normal.pdf(X, mean=mean, cov=cov, allow_singular=True)  # <--- fix added here
            elif self.mode == 'naive':
                means, stds = self.class_params[cls]
                likelihood = np.prod(norm.pdf(X, loc=means, scale=stds), axis=1)
            elif self.mode == 'nonparametric':
                kde = self.class_params[cls]
                likelihood = np.exp(kde.score_samples(X))
            probs[:, idx] = prior * likelihood
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]