"""
 PROJECT 2  |  DATA CLASSIFICATION USING AI
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # render plots to files (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# Setup: reproducibility + output folder + consistent plot style
RANDOM_STATE = 42                      # fixed seed => results reproducible
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")


def banner(title):
    """Pretty section header so the console output reads like a report."""
    line = "=" * 74
    print(f"\n{line}\n {title}\n{line}")



# STEP 1 — LOAD AND UNDERSTAND THE DATASET
def load_and_understand():
    banner("STEP 1  |  LOAD AND UNDERSTAND THE DATASET")

    # --- Load 
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # Add a human-readable species column next to the numeric target
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

    feature_names = iris.feature_names
    target_names = list(iris.target_names)

    # --- Understand: shape, preview, stats, class balance, missing values ---
    print(f"\nDataset shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Feature columns (X)    : {feature_names}")
    print(f"Target classes (y)     : {target_names}")

    print("\nFirst 5 samples:")
    print(df.head().to_string(index=False))

    print("\nStatistical summary (describe):")
    print(df[feature_names].describe().round(2).to_string())

    print("\nClass distribution (is the data balanced?):")
    print(df["species"].value_counts().to_string())

    print("\nMissing values per column:")
    print(df[feature_names].isnull().sum().to_string())
    print("\n-> No missing values: the dataset is clean and ready to use.")

    # --- Understand visually: class balance + feature relationships ---------
    _plot_class_distribution(df)
    _plot_feature_pairplot(df, feature_names)
    _plot_correlation_heatmap(df, feature_names)

    return df, feature_names, target_names


def _plot_class_distribution(df):
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="species", hue="species", legend=False)
    for container in ax.containers:
        ax.bar_label(container)
    ax.set_title("Class Distribution — 50 samples per species (balanced)")
    ax.set_xlabel("Species")
    ax.set_ylabel("Number of samples")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_class_distribution.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   [saved] {os.path.basename(path)}")


def _plot_feature_pairplot(df, feature_names):
    g = sns.pairplot(df, vars=feature_names, hue="species", diag_kind="hist",
                     height=1.8, plot_kws={"alpha": 0.7, "s": 25})
    g.figure.suptitle("Feature Relationships by Species", y=1.02)
    path = os.path.join(OUTPUT_DIR, "02_feature_pairplot.png")
    g.savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"   [saved] {os.path.basename(path)}")


def _plot_correlation_heatmap(df, feature_names):
    plt.figure(figsize=(7, 5.5))
    corr = df[feature_names].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True,
                cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_correlation_heatmap.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   [saved] {os.path.basename(path)}")


# STEP 2 — SPLIT DATA INTO TRAINING AND TESTING SETS
def split_data(df, feature_names):
    banner("STEP 2  |  SPLIT DATA INTO TRAINING AND TESTING SETS")

    X = df[feature_names].values        # inputs  (what the model sees)
    y = df["target"].values             # labels  (what the model must predict)

    # 80% train / 20% test.
    # stratify=y  -> keep the same class ratio in both sets.
    # random_state -> the same split every run (reproducible).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTotal samples          : {len(X)}")
    print(f"Training samples (80%) : {len(X_train)}")
    print(f"Testing  samples (20%) : {len(X_test)}")
    print("\nThe model learns ONLY from the training set.")
    print("The testing set is kept hidden until the final evaluation —")
    print("this is how we check the model generalises to unseen data.")

    # Feature scaling: put all 4 features on the same scale.
    # Fit the scaler on TRAIN only, then apply to both (no data leakage).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nFeatures standardised (mean 0, std 1) using the training set only.")

    return X_train_scaled, X_test_scaled, y_train, y_test


# STEP 3 — APPLY A CLASSIFICATION ALGORITHM (+ compare several)
def train_and_compare(X_train, X_test, y_train, y_test, target_names):
    banner("STEP 3  |  APPLY CLASSIFICATION ALGORITHMS & TRAIN")

    # A small line-up of simple, classic classifiers to compare.
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM (linear)":        SVC(kernel="linear", random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        # TRAIN on the training set
        model.fit(X_train, y_train)

        # VALIDATE with 5-fold cross-validation on the training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        # TEST on the held-out test set
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_acc": test_acc,
            "y_pred": y_pred,
        }
        print(f"\n{name}")
        print(f"   Cross-val accuracy (train) : "
              f"{cv_scores.mean()*100:5.2f}%  (+/- {cv_scores.std()*100:.2f}%)")
        print(f"   Test accuracy (unseen)     : {test_acc*100:5.2f}%")

    # Pick the winner by test accuracy (ties broken by CV mean)
    best_name = max(results, key=lambda n: (results[n]["test_acc"],
                                            results[n]["cv_mean"]))
    print(f"\n>> Best model: {best_name} "
          f"({results[best_name]['test_acc']*100:.2f}% test accuracy)")

    _plot_model_comparison(results)
    return results, best_name


def _plot_model_comparison(results):
    names = list(results.keys())
    cv = [results[n]["cv_mean"] * 100 for n in names]
    test = [results[n]["test_acc"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.38
    plt.figure(figsize=(9, 5.5))
    plt.bar(x - width/2, cv, width, label="Cross-val accuracy (train)")
    plt.bar(x + width/2, test, width, label="Test accuracy (unseen)")
    for i, (c, t) in enumerate(zip(cv, test)):
        plt.text(i - width/2, c + 0.4, f"{c:.1f}", ha="center", fontsize=8)
        plt.text(i + width/2, t + 0.4, f"{t:.1f}", ha="center", fontsize=8)
    plt.xticks(x, names, rotation=15)
    plt.ylim(80, 103)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison — Cross-validation vs. Test Accuracy")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_model_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"   [saved] {os.path.basename(path)}")


# STEP 4 — VALIDATE THE BEST MODEL IN DETAIL
def validate_best(results, best_name, y_test, target_names):
    banner(f"STEP 4  |  VALIDATE THE BEST MODEL  ->  {best_name}")

    y_pred = results[best_name]["y_pred"]

    print("\nClassification report (precision / recall / f1 per class):\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=target_names, columns=target_names).to_string())

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={"shrink": 0.8})
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted species")
    plt.ylabel("Actual species")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n   [saved] {os.path.basename(path)}")


# STEP 5 — TEST THE MODEL ON COMPLETELY NEW DATA
#          (suggested in the project conclusion)
def test_on_new_data(df, feature_names, target_names):
    banner("STEP 5  |  TEST THE MODEL ON BRAND-NEW, MADE-UP FLOWERS")

    # Re-train the chosen model (KNN) on the WHOLE dataset so it uses every
    # available example before facing genuinely new measurements.
    X = df[feature_names].values
    y = df["target"].values
    scaler = StandardScaler().fit(X)
    final_model = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(X), y)

    # Three hand-written flowers the model has never seen.
    # Order of features: sepal length, sepal width, petal length, petal width (cm)
    new_flowers = np.array([
        [5.0, 3.5, 1.4, 0.2],   # small petals  -> expect setosa
        [6.0, 2.7, 4.5, 1.5],   # medium petals -> expect versicolor
        [6.7, 3.1, 5.6, 2.4],   # large petals  -> expect virginica
    ])
    labels = ["Flower A (tiny petals)",
              "Flower B (medium petals)",
              "Flower C (large petals)"]

    preds = final_model.predict(scaler.transform(new_flowers))
    probs_available = hasattr(final_model, "predict_proba")

    print("\nFeeding 3 new flowers to the trained model:\n")
    for lbl, measures, p in zip(labels, new_flowers, preds):
        print(f"   {lbl}")
        print(f"      measurements : {measures.tolist()} cm")
        print(f"      prediction   : {target_names[p].upper()}")
        print()

    print("The model confidently sorts each new flower into a species using")
    print("only what it learned from the training data — supervised learning "
          "in action.")


# MAIN
def main():
    banner("PROJECT 2  |  DATA CLASSIFICATION USING AI  |  DecodeLabs")
    print("Pipeline:  Load -> Understand -> Split -> Train -> Validate -> Predict")

    df, feature_names, target_names = load_and_understand()
    X_train, X_test, y_train, y_test = split_data(df, feature_names)
    results, best_name = train_and_compare(
        X_train, X_test, y_train, y_test, target_names)
    validate_best(results, best_name, y_test, target_names)
    test_on_new_data(df, feature_names, target_names)

    banner("PROJECT 2 COMPLETE  ✔")
    print("All requirements met:")
    print("  [x] Loaded and understood the dataset (EDA + visualisations)")
    print("  [x] Split data into training and testing sets (80/20, stratified)")
    print("  [x] Applied classification algorithms (KNN, Tree, LogReg, SVM)")
    print("  [x] Trained, tested, and validated the model")
    print("  [x] Bonus: compared algorithms + tested on brand-new data")
    print(f"\nAll charts saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()