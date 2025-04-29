
import numpy as np

class CustomPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigen_vals)[::-1]
        self.components_ = eigen_vecs[:, sorted_indices[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

class CustomSFFS:
    def __init__(self, estimator, k_features, scoring='accuracy', cv=5):
        self.estimator = estimator
        self.k_features = k_features
        self.scoring = scoring
        self.cv = cv
        self.indices_ = []

    def fit(self, X, y):
        n_features = X.shape[1]
        remaining = set(range(n_features))
        selected = []

        while len(selected) < self.k_features:
            best_score = -np.inf
            best_feature = None
            for feature in remaining:
                trial_features = selected + [feature]
                score = np.mean(cross_val_score(clone(self.estimator), X[:, trial_features], y, cv=self.cv, scoring=self.scoring))
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            if best_feature is None:
                break 

            selected.append(best_feature)
            remaining.remove(best_feature)

            improved = True
            while improved and len(selected) > 2:
                improved = False
                for feature in selected[:-1]:
                    trial_features = [f for f in selected if f != feature]
                    score = np.mean(cross_val_score(clone(self.estimator), X[:, trial_features], y, cv=self.cv, scoring=self.scoring))
                    if score > best_score:
                        best_score = score
                        selected.remove(feature)
                        improved = True
                        break

        self.indices_ = selected

    def transform(self, X):
        return X[:, self.indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
