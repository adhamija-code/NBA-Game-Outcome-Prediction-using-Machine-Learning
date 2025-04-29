from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_models():
    return {
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
        "Random Forest": lambda: RandomForestClassifier(),
        "SVM": lambda: SVC(probability=True)
    }
