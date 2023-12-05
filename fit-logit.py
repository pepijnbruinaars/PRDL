import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp


def fit_logit(param):
    test_digits = pd.read_csv("data/test_digits.csv").values
    test_labels = pd.read_csv("data/test_labels.csv").values
    train_digits = pd.read_csv("data/train_digits.csv").values
    train_labels = pd.read_csv("data/train_labels.csv").values
    logit = LogisticRegression(
        C=param,
        penalty="l1",
        solver="saga",
        multi_class="multinomial",
        max_iter=300,
        verbose=2,
    )
    logit.fit(train_digits, train_labels.ravel())
    score = logit.score(test_digits, test_labels)
    print("done fitting param: ", param, " score: ", score)
    return score


def main():
    # Fit multinomial logistic regression with LASSO penalty
    # Use cross-validation to select regularization parameter
    # Use all cores available
    param_space = np.logspace(-4, 0, 30)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        scores = pool.map(fit_logit, param_space)

        # save scores
        np.savetxt("multinomial-scores.csv", scores, delimiter=",")

    # Plot scores
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(param_space, scores)
    plt.xscale("log")
    plt.xlabel("Regularization parameter")
    plt.ylabel("Accuracy")
    plt.title("Multinomial logistic regression")
    plt.tight_layout()
    plt.savefig("multinomial-lasso.png")


if __name__ == "__main__":
    main()
