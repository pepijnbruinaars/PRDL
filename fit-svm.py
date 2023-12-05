import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import multiprocessing as mp


def fit_SVC(param):
    test_digits = pd.read_csv("data/test_digits.csv").values
    test_labels = pd.read_csv("data/test_labels.csv").values
    train_digits = pd.read_csv("data/train_digits.csv").values
    train_labels = pd.read_csv("data/train_labels.csv").values

    new_svc = SVC(
        C=param,
        kernel="linear",
        degree=2,
    )

    new_svc.fit(train_digits, train_labels.ravel())
    score = new_svc.score(test_digits, test_labels)
    print("done fitting param: ", param, " score: ", score)
    return score


def main():
    # Fit multinomial logistic regression with LASSO penalty
    # Use cross-validation to select regularization parameter
    # Use all cores available
    param_space = np.logspace(-8, -4, 50)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        linear_scores = pool.map(fit_SVC, param_space)

        # save scores
        np.savetxt("svc-linear-2-scores.csv", linear_scores, delimiter=",")

    # Plot scores
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(param_space, linear_scores)
    plt.xscale("log")
    plt.xlabel("Regularization parameter")
    plt.ylabel("Accuracy")
    plt.title("Multinomial logistic regression")
    plt.tight_layout()
    plt.savefig("svm.png")


if __name__ == "__main__":
    main()
