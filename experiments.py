import numpy as np
import preprocess, evaluate

def run_repeated_experiment(X, y, model_func, runs=5):
    accuracies = []
    for i in range(runs):
        X_train, X_test, y_train, y_test = preprocess.split_data(X, y)
        model = model_func()
        model.fit(X_train, y_train)
        acc, cm, report = evaluate.evaluate_model(model, X_test, y_test)
        accuracies.append(acc)
        print(f"Run {i+1} Accuracy: {acc:.4f}")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Deviation: {np.std(accuracies):.4f}")
    return accuracies

def cumulative_confusion_matrix(model_func, X, y, class_names, runs=10):
    total_cm = np.zeros((2, 2))
    for _ in range(runs):
        X_train, X_test, y_train, y_test = preprocess.split_data(X, y)
        model = model_func()
        model.fit(X_train, y_train)
        _, cm, _ = evaluate.evaluate_model(model, X_test, y_test)
        total_cm += cm
    evaluate.plot_confusion_matrix(total_cm, class_names, title="Cumulative Confusion Matrix")
