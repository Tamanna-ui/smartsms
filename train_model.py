import json
import os
import re
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smart_sms_shield_mpl"))

import matplotlib.pyplot as plt


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text))
    words = text.lower().split()
    return " ".join(word for word in words if word not in ENGLISH_STOP_WORDS)


def load_dataset() -> pd.DataFrame:
    candidate_files = [
        DATA_DIR / "spam.csv",
        DATA_DIR / "SMSSpamCollection.csv",
        DATA_DIR / "sms_spam_sample.csv",
    ]

    for file_path in candidate_files:
        if file_path.exists():
            dataframe = pd.read_csv(file_path, encoding="latin-1")
            break
    else:
        raise FileNotFoundError("No dataset found in the data directory.")

    dataframe = dataframe.rename(
        columns={
            "v1": "label",
            "v2": "message",
            "Category": "label",
            "Message": "message",
        }
    )

    if not {"label", "message"}.issubset(dataframe.columns):
        raise ValueError("Dataset must contain label/message or v1/v2 columns.")

    dataframe = dataframe[["label", "message"]].dropna()
    dataframe["label"] = dataframe["label"].astype(str).str.strip().str.lower()
    dataframe["message"] = dataframe["message"].astype(str).map(clean_text)
    return dataframe[dataframe["label"].isin(["ham", "spam"])]


def build_models() -> dict:
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Linear SVM": LinearSVC(),
    }


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True),
            ),
            ("classifier", model),
        ]
    )


def save_confusion_matrix(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["ham", "spam"])
    figure, axis = plt.subplots(figsize=(6, 5))
    display.plot(ax=axis, cmap="YlGnBu", colorbar=False)
    axis.set_title("Spam Classifier Confusion Matrix")
    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close(figure)


def save_accuracy_chart(results: dict):
    names = list(results.keys())
    accuracies = [result["accuracy"] for result in results.values()]

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(names, accuracies, color=["#0f766e", "#f59e0b", "#b45309"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.set_title("Model Accuracy Comparison")

    for bar, accuracy in zip(bars, accuracies):
        axis.text(bar.get_x() + bar.get_width() / 2, accuracy + 0.02, f"{accuracy:.2%}", ha="center")

    figure.tight_layout()
    figure.savefig(ARTIFACTS_DIR / "model_comparison.png", dpi=200)
    plt.close(figure)


def main():
    dataframe = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe["message"],
        dataframe["label"],
        test_size=0.2,
        random_state=42,
        stratify=dataframe["label"],
    )

    best_name = ""
    best_accuracy = -1.0
    best_pipeline = None
    best_predictions = None
    results = {}

    for model_name, model in build_models().items():
        pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
        accuracy = accuracy_score(y_test, predictions)
        results[model_name] = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1_score), 4),
        }

        if accuracy > best_accuracy:
            best_name = model_name
            best_accuracy = accuracy
            best_pipeline = pipeline
            best_predictions = predictions

    save_confusion_matrix(y_test, best_predictions)
    save_accuracy_chart(results)

    metrics = {
        "dataset_size": int(len(dataframe)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "best_model": best_name,
        "results": results,
        "classification_report": classification_report(
            y_test,
            best_predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    with (ARTIFACTS_DIR / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    joblib.dump(best_pipeline, ARTIFACTS_DIR / "spam_classifier.joblib")
    print(f"Best model: {best_name}")
    print(f"Accuracy: {best_accuracy:.2%}")
    print(f"Artifacts saved in: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
