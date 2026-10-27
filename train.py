from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

import tldextract
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import URL_FEATURES, extract_url_features


def load_real_data(csv_path="data/urls.csv"):
    data = pd.read_csv(csv_path)

    required = {"url", "label"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain columns named 'url' and 'label'.")

    # Accept common label spellings and convert them to the model's labels:
    # 0 = legitimate, 1 = phishing.
    label_map = {
        "legitimate": 0,
        "benign": 0,
        "good": 0,
        "0": 0,
        "phishing": 1,
        "phish": 1,
        "malicious": 1,
        "bad": 1,
        "1": 1,
    }

    data = data.dropna(subset=["url", "label"]).copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["target"] = data["label"].map(label_map)

    if data["target"].isna().any():
        unknown = sorted(data.loc[data["target"].isna(), "label"].unique())
        raise ValueError(f"Unrecognized labels in CSV: {unknown}")

    # Remove duplicate URLs so copies don't appear in both train and test data.
    data["url"] = data["url"].astype(str).str.strip()
    data = data.drop_duplicates(subset=["url"])

    rows = []
    groups = []
    valid_indices = []

    for index, url in data["url"].items():
        try:
            rows.append(extract_url_features(url))

            # Keep subdomains of the same registered domain in one group.
            domain = tldextract.extract(url).top_domain_under_public_suffix
            groups.append(domain or url)

            valid_indices.append(index)
        except ValueError:
            # Skip rows that aren't valid URLs for this feature extractor.
            continue

    X = pd.DataFrame(rows, columns=URL_FEATURES)
    y = data.loc[valid_indices, "target"].astype(int).to_numpy()

    if len(set(y)) < 2:
        raise ValueError("The dataset must contain both legitimate and phishing URLs.")

    return X, y, groups

def main():
    X, y, groups = load_real_data()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    train_domains = {groups[i] for i in train_indices}
    test_domains = {groups[i] for i in test_indices}
    overlap = train_domains & test_domains

    print(f"Train rows: {len(train_indices):,}")
    print(f"Test rows: {len(test_indices):,}")
    print(f"Domains shared between train and test: {len(overlap)}")

    assert not overlap, "Domain leakage detected between train and test."

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "confusion_matrix": confusion_matrix(
                y_test, predictions
            ).tolist(),
        })

        print(f"\n{name}")
        print(classification_report(y_test, predictions))

    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    joblib.dump(models["Random Forest"], output_dir / "model.joblib")
    joblib.dump(
        {
            "feature_names": URL_FEATURES,
            "metrics": results,
            "label_0": "legitimate",
            "label_1": "phishing",
        },
        output_dir / "metadata.joblib",
    )

    pd.DataFrame([
        {
            key: value
            for key, value in row.items()
            if key != "confusion_matrix"
        }
        for row in results
    ]).to_csv(output_dir / "metrics.csv", index=False)

    print("\nSaved model and metrics in the artifacts folder.")


if __name__ == "__main__":
    main()