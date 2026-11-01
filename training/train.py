"""Train breast-cancer triage models, select by CV malignant recall, and save artifacts."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "target"
MALIGNANT = 0  # in this dataset 0 = malignant (the positive class), 1 = benign
LABELS = {0: "malignant", 1: "benign"}


def malignant_probability(model, X):
    """Probability of the malignant class, whatever order the model stores its classes in."""
    malignant_index = list(model.classes_).index(MALIGNANT)
    return model.predict_proba(X)[:, malignant_index]


def malignant_recall(model, X, y):
    return recall_score(y, model.predict(X), pos_label=MALIGNANT)


def accuracy(model, X, y):
    return accuracy_score(y, model.predict(X))


def malignant_roc_auc(model, X, y):
    return roc_auc_score((y == MALIGNANT).astype(int), malignant_probability(model, X))


SCORERS = {
    "malignant_recall": malignant_recall,
    "accuracy": accuracy,
    "roc_auc": malignant_roc_auc,
}


def build_candidates(params, seed):
    """Each candidate is one Pipeline, so preprocessing travels with the model."""
    lr = params["models"]["logistic_regression"]
    rf = params["models"]["random_forest"]
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=lr["C"], max_iter=lr["max_iter"], class_weight=lr["class_weight"],
                random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=rf["n_estimators"], max_depth=rf["max_depth"],
                min_samples_leaf=rf["min_samples_leaf"], class_weight=rf["class_weight"],
                random_state=seed, n_jobs=1)),
        ]),
    }


def evaluate(model, X, y):
    return {name: round(float(scorer(model, X, y)), 4) for name, scorer in SCORERS.items()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-version", default="1.0.0")
    parser.add_argument("--metrics-path", type=Path, default=Path("metrics.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    params = yaml.safe_load(args.params.read_text())
    seed = params["seed"]
    data_path = args.data_path or Path(params["data"]["path"])

    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["data"]["test_size"], stratify=y, random_state=seed)
    print(f"Data: {len(df)} rows from {data_path} | train={len(X_train)} test={len(X_test)}")

    cv = StratifiedKFold(n_splits=params["cv"]["folds"], shuffle=True, random_state=seed)
    results = {}
    for name, model in build_candidates(params, seed).items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORERS)
        cv_metrics = {k: round(float(scores[f"test_{k}"].mean()), 4) for k in SCORERS}
        model.fit(X_train, y_train)
        test_metrics = evaluate(model, X_test, y_test)
        results[name] = {"cv": cv_metrics, "test": test_metrics, "model": model}
        print(f"{name:20s} CV {cv_metrics} | TEST {test_metrics}")

    best_name = max(
        results,
        key=lambda n: (results[n]["cv"]["malignant_recall"], results[n]["cv"]["roc_auc"]))
    best = results[best_name]
    print(f"Selected: {best_name} (highest mean CV malignant recall)")

    min_recall = params["selection"]["min_test_malignant_recall"]
    if best["test"]["malignant_recall"] < min_recall:
        print(f"FAIL: test malignant recall {best['test']['malignant_recall']} "
              f"is below {min_recall}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], args.output_dir / "model.joblib")

    candidates_summary = {n: {"cv": r["cv"], "test": r["test"]} for n, r in results.items()}
    model_card = {
        "model_version": args.model_version,
        "model_family": best_name,
        "feature_names": list(X.columns),
        "n_features": X.shape[1],
        "positive_class": "malignant",
        "label_mapping": LABELS,
        "metrics": {"cv": best["cv"], "test": best["test"]},
        "candidates": candidates_summary,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "data": {"path": str(data_path), "rows": len(df),
                 "train_rows": len(X_train), "test_rows": len(X_test)},
    }
    (args.output_dir / "model_card.json").write_text(json.dumps(model_card, indent=2))

    metrics = {"selected_model": best_name, "test": best["test"], "cv": best["cv"],
               "candidates": candidates_summary}
    args.metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved {args.output_dir}/model.joblib, model_card.json and {args.metrics_path}")


if __name__ == "__main__":
    main()
