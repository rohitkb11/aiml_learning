
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# -----------------------------
# 1. Load and prepare the data
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASS_NAMES = ["Low", "High"]

df = pd.read_csv("Admission_Predict.xls")
df.columns = df.columns.str.strip()

df = df.drop(columns=["Serial No."])

# Convert admission chance into Low (0) and High (1)
threshold = df["Chance of Admit"].quantile(0.75)
df["Chance of Admit"] = (
    df["Chance of Admit"] >= threshold
).astype(int)

X = df.drop(columns=["Chance of Admit"])
y = df["Chance of Admit"]

# -----------------------------
# 2. Split the dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Admission chance threshold: {threshold:.4f}")

# -----------------------------
# 3. Evaluation function
# -----------------------------
def evaluate_model(model, model_name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = {
        "Accuracy": accuracy_score(y_train, y_train_pred),
        "Precision": precision_score(
            y_train, y_train_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_train, y_train_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_train, y_train_pred, zero_division=0
        )
    }

    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(
            y_test, y_test_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_test_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_test_pred, zero_division=0
        )
    }

    print(f"\n{model_name}")
    print("\nTraining Metrics:")
    print(pd.DataFrame([train_metrics], index=["Training"]).round(2))

    print("\nTesting Metrics:")
    print(pd.DataFrame([test_metrics], index=["Testing"]).round(2))

    return train_metrics, test_metrics, y_test_pred


# -----------------------------
# 4. Confusion matrix function
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 5. Decision Tree - Entropy
# -----------------------------
dt_entropy = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

dt_entropy.fit(X_train, y_train)

entropy_train, entropy_test, y_pred_entropy = evaluate_model(
    dt_entropy, "Decision Tree - Entropy"
)

plot_confusion_matrix(
    y_test, y_pred_entropy, "Decision Tree - Entropy"
)

plt.figure(figsize=(15, 10))
plot_tree(
    dt_entropy,
    feature_names=X_train.columns,
    class_names=CLASS_NAMES,
    filled=True
)
plt.title("Decision Tree - Entropy")
plt.tight_layout()
plt.show()


# -----------------------------
# 6. Decision Tree - Gini
# -----------------------------
dt_gini = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

dt_gini.fit(X_train, y_train)

gini_train, gini_test, y_pred_gini = evaluate_model(
    dt_gini, "Decision Tree - Gini"
)

plot_confusion_matrix(
    y_test, y_pred_gini, "Decision Tree - Gini"
)

plt.figure(figsize=(15, 10))
plot_tree(
    dt_gini,
    feature_names=X_train.columns,
    class_names=CLASS_NAMES,
    filled=True
)
plt.title("Decision Tree - Gini")
plt.tight_layout()
plt.show()


# -----------------------------
# 7. Gaussian Naive Bayes
# -----------------------------
nb = GaussianNB(
    priors=None,
    var_smoothing=1e-9
)

nb.fit(X_train, y_train)

nb_train, nb_test, y_pred_nb = evaluate_model(
    nb, "Gaussian Naive Bayes"
)

plot_confusion_matrix(
    y_test, y_pred_nb, "Gaussian Naive Bayes"
)


# -----------------------------
# 8. Compare model performance
# -----------------------------
results = pd.DataFrame(
    [entropy_test, gini_test, nb_test],
    index=["Entropy", "Gini", "Naive Bayes"]
)

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

ax = results[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)

plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()


# -----------------------------
# 9. Predict for a new student
# -----------------------------
def predict_student():
    print("\nEnter Student Details")
    print("-" * 30)

    gre = float(input("GRE Score (290-340): "))
    toefl = float(input("TOEFL Score (92-120): "))
    university_rating = float(input("University Rating (1-5): "))
    sop = float(input("SOP (1-5): "))
    lor = float(input("LOR (1-5): "))
    cgpa = float(input("CGPA (6.8-9.92): "))
    research = int(input("Research (0 = No, 1 = Yes): "))

    ranges = {
        "GRE Score": (gre, 290, 340),
        "TOEFL Score": (toefl, 92, 120),
        "University Rating": (university_rating, 1, 5),
        "SOP": (sop, 1, 5),
        "LOR": (lor, 1, 5),
        "CGPA": (cgpa, 6.8, 9.92),
        "Research": (research, 0, 1)
    }

    for name, (value, minimum, maximum) in ranges.items():
        if not minimum <= value <= maximum:
            print(f"Error: {name} must be between {minimum} and {maximum}.")
            return

    if research not in [0, 1]:
        print("Error: Research must be 0 or 1.")
        return

    student = pd.DataFrame([{
        "GRE Score": gre,
        "TOEFL Score": toefl,
        "University Rating": university_rating,
        "SOP": sop,
        "LOR": lor,
        "CGPA": cgpa,
        "Research": research
    }])

    predictions = {
        "Decision Tree (Entropy)": dt_entropy.predict(student)[0],
        "Decision Tree (Gini)": dt_gini.predict(student)[0],
        "Naive Bayes": nb.predict(student)[0]
    }

    result = pd.DataFrame({
        "Model": predictions.keys(),
        "Prediction": [
            CLASS_NAMES[value] for value in predictions.values()
        ]
    })

    print("\nPrediction Results:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    predict_student()
