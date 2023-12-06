import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp

from sklearn.model_selection import cross_val_score


def fit_logit(param):
    train_digits = pd.read_csv("data/train_digits.csv").values
    train_labels = pd.read_csv("data/train_labels.csv").values
    logit = LogisticRegression(
        C=param,
        penalty="l1",
        solver="saga",
        multi_class="multinomial",
        max_iter=500,
        verbose=2,
    )
    # Get cross-validated accuracy, recall and precision
    scores = cross_val_score(logit, train_digits, train_labels.ravel(), cv=5)

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    return mean_score, std_score


def main():
    # Fit multinomial logistic regression with LASSO penalty
    # Use cross-validation to select regularization parameter
    # Use all cores available
    param_space = np.logspace(-4, 0, 50)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        scores = pool.map(fit_logit, param_space)

        (
            mean_accuracy,
            std_accuracy,
        ) = zip(*scores)
        dataframe = pd.DataFrame(
            {
                "param": param_space,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
            }
        )
        # save scores
        dataframe.to_csv("logit-lasso-scores.csv", index=False)

    # Plot scores
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(param_space, mean_accuracy)

    # Plot error bars showing standard deviation with fill
    plt.fill_between(
        param_space,
        np.array(mean_accuracy) - np.array(std_accuracy),
        np.array(mean_accuracy) + np.array(std_accuracy),
        alpha=0.2,
    )
    plt.xscale("log")
    plt.xlabel("Regularization parameter")
    plt.ylabel("Accuracy")
    plt.title("Multinomial logistic regression")
    plt.tight_layout()
    plt.savefig("multinomial-lasso.png")


if __name__ == "__main__":
    main()
