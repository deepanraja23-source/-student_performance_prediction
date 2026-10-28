"""Student Performance Prediction (Pass / Fail) - Summer Internship Project
Usage:
  python student_prediction.py train
  python student_prediction.py predict --attendance 80 --study_hours 3 --previous_marks 65 --assignment_marks 70 --internal_marks 60
"""
import argparse, sys
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ---------------- Configuration (edit to match your dataset) ----------------
DATA_PATH = "data/students.csv"
MODEL_PATH = "outputs/best_model.joblib"
TARGET = "result"
FEATURES = ["attendance", "study_hours", "previous_marks", "assignment_marks", "internal_marks"]
LIMITS = {"attendance": (0, 100), "study_hours": (0, 24), "previous_marks": (0, 100),
          "assignment_marks": (0, 100), "internal_marks": (0, 100)}
TEST_SIZE, SEED = 0.2, 42


# ---------------- Data loading and cleaning ----------------
def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {path}")
    except pd.errors.EmptyDataError:
        raise ValueError("Dataset file is empty")
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df


def clean_data(df):
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    return df


# ---------------- Exploratory analysis ----------------
def explore(df):
    print("\nDataset shape:", df.shape)
    print(df.describe().round(2))
    print("\nClass counts:\n", df[TARGET].value_counts())
    df[FEATURES].hist(figsize=(10, 7), bins=20)
    plt.tight_layout(); plt.savefig("outputs/feature_histograms.png"); plt.close()
    corr = df[FEATURES].corr()
    plt.figure(figsize=(6, 5)); plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1); plt.colorbar()
    plt.xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right"); plt.yticks(range(len(FEATURES)), FEATURES)
    plt.title("Correlation of features"); plt.tight_layout(); plt.savefig("outputs/correlation.png"); plt.close()


# ---------------- Training and evaluation ----------------
def get_models():
    return {
        "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
        "Decision Tree": Pipeline([("m", DecisionTreeClassifier(max_depth=5, random_state=SEED))]),
        "Random Forest": Pipeline([("m", RandomForestClassifier(n_estimators=200, random_state=SEED))]),
    }


def train_and_evaluate(df):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    rows, fitted = [], {}
    for name, model in get_models().items():
        cv = cross_val_score(model, X_tr, y_tr, cv=5).mean()
        model.fit(X_tr, y_tr)
        p = model.predict(X_te)
        rows.append({"Model": name, "CV accuracy": cv, "Accuracy": accuracy_score(y_te, p),
                     "Precision": precision_score(y_te, p, pos_label="Pass"),
                     "Recall": recall_score(y_te, p, pos_label="Pass"),
                     "F1": f1_score(y_te, p, pos_label="Pass")})
        fitted[name] = (model, p)
    res = pd.DataFrame(rows).set_index("Model").round(3)
    print("\nModel comparison (test set):\n", res)
    res.to_csv("outputs/model_comparison.csv")
    best = res["Accuracy"].idxmax()
    model, pred = fitted[best]
    print(f"\nBest model: {best}\n")
    print("Confusion matrix:\n", confusion_matrix(y_te, pred, labels=["Pass", "Fail"]))
    print(classification_report(y_te, pred))
    est = model.named_steps["m"]
    if hasattr(est, "feature_importances_"):
        s = pd.Series(est.feature_importances_, index=FEATURES).sort_values()
        s.plot(kind="barh", title=f"Feature importance ({best})"); plt.tight_layout()
        plt.savefig("outputs/feature_importance.png"); plt.close()
    joblib.dump(model, MODEL_PATH)
    print("Saved model to", MODEL_PATH)


# ---------------- Prediction ----------------
def predict_student(values):
    """values: dict of feature -> number. Returns 'Pass' or 'Fail'."""
    for f in FEATURES:
        v = values.get(f)
        if v is None or str(v).strip() == "":
            raise ValueError(f"Please enter a value for '{f}'")
        try:
            v = float(v)
        except ValueError:
            raise ValueError(f"'{f}' must be a number")
        lo, hi = LIMITS[f]
        if not lo <= v <= hi:
            raise ValueError(f"'{f}' must be between {lo} and {hi}")
        values[f] = v
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError("Trained model not found. Run: python student_prediction.py train")
    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    label = model.predict(X)[0]
    prob = dict(zip(model.classes_, model.predict_proba(X)[0])) if hasattr(model, "predict_proba") else {}
    return label, prob


def main():
    ap = argparse.ArgumentParser(description="Student Performance Prediction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="clean data, train, compare and save the best model")
    pr = sub.add_parser("predict", help="predict for one student")
    for f in FEATURES:
        pr.add_argument(f"--{f}", required=True)
    a = ap.parse_args()
    try:
        if a.cmd == "train":
            df = clean_data(load_data()); explore(df); train_and_evaluate(df)
        else:
            label, prob = predict_student({f: getattr(a, f) for f in FEATURES})
            print("Predicted result:", label)
            for k, v in prob.items():
                print(f"  P({k}) = {v:.2f}")
    except (FileNotFoundError, ValueError) as e:
        print("Error:", e); sys.exit(1)


if __name__ == "__main__":
    main()
