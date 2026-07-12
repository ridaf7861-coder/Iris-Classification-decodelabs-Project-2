# Iris Classification — Data Classification Using AI

> **DecodeLabs Industrial Training Kit · Project 2 · Batch 2026**

A supervised machine learning project that trains an AI model to classify Iris
flowers into their correct species. It walks through the complete
classification pipeline — **Load → Understand → Split → Train → Validate → Predict** —
and compares four classic algorithms, with the linear SVM reaching **100% test accuracy**.

---

## 🎯 Overview

The goal is to teach a machine to recognise patterns in flower measurements and
categorise a flower into one of three species. This is a textbook **supervised
learning** classification task and the perfect first model for any aspiring AI engineer.

**Dataset:** The classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
— 150 samples, 4 numeric features (sepal length, sepal width, petal length,
petal width) and 3 balanced classes (*setosa*, *versicolor*, *virginica*).

---

## ⚙️ How to run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/iris-classification-decodelabs.git
cd iris-classification-decodelabs

# 2. (Optional) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python data_classification.py
```

The script prints a full report to the console and saves 5 charts into the `outputs/` folder.

---

## 🧪 The pipeline, step by step

| Step | What happens |
|------|--------------|
| **1. Load & Understand** | Loads Iris, prints shape/preview/statistics/class balance, checks for missing values, and generates exploratory charts |
| **2. Split** | 80% training / 20% testing, stratified to keep class ratios, with feature scaling fit on the training set only (no data leakage) |
| **3. Train & Compare** | Trains KNN, Decision Tree, Logistic Regression, and a linear SVM; validates each with 5-fold cross-validation |
| **4. Validate** | Inspects the best model with a precision/recall/F1 report and a confusion matrix |
| **5. Predict** | Feeds the model 3 brand-new, unseen flowers to prove it generalises |

---

## 📊 Results

| Model | Cross-val accuracy (train) | Test accuracy (unseen) |
|-------|:--------------------------:|:----------------------:|
| K-Nearest Neighbors | 96.67% | 93.33% |
| Decision Tree       | 94.17% | 93.33% |
| Logistic Regression | 95.83% | 93.33% |
| **SVM (linear)**    | **97.50%** | **100.00%** 🏆 |

The linear SVM classified all 30 unseen test flowers correctly and correctly
labelled all 3 brand-new made-up flowers.

---

## 📈 Output charts

All saved to the `outputs/` folder when you run the script:

1. `01_class_distribution.png` — confirms the 3 classes are balanced
2. `02_feature_pairplot.png` — how the species separate across features
3. `03_correlation_heatmap.png` — relationships between the 4 features
4. `04_model_comparison.png` — accuracy of all 4 algorithms
5. `05_confusion_matrix.png` — detailed correctness of the best model

---

## 🛠️ Tech stack

**Python** · **scikit-learn** · **pandas** · **NumPy** · **Matplotlib** · **Seaborn**

---

## 🧠 Skills demonstrated

Data handling · exploratory data analysis · train/test splitting ·
supervised learning · model training · cross-validation · model evaluation.

---

## 📄 License

Free to use for learning and portfolio purposes.
