import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.base import clone

from synthtree import SynthTree

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# sklearn-specific:
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_emotion_csv(path: Path, target_col: str = "emotion_label"):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path.name}")

    X = df.drop(columns=[target_col]).values.astype(float)
    y_raw = df[target_col].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    feature_names = df.drop(columns=[target_col]).columns.tolist()
    class_names = le.classes_.tolist()
    return X, y, feature_names, class_names, le


# ---------------------------------------------------------
# Black-box models
# ---------------------------------------------------------

def get_black_box_models(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    5 different black-box models.
    Each wrapped in a pipeline with StandardScaler where appropriate.
    """
    models = {
        "RF": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GBM": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=3.0,
                gamma="scale",
                kernel="rbf",
                probability=True,
                random_state=random_state,
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size="auto",
                learning_rate="adaptive",
                max_iter=300,
                random_state=random_state,
            )),
        ]),
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                multi_class="auto",
                solver="lbfgs",
                max_iter=2000,
                random_state=random_state,
            )),
        ]),
    }
    return models


# ---------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------

def evaluate_models_per_class(
    X: np.ndarray,
    y: np.ndarray,
    model_prototypes: Dict[str, object],
    n_runs: int = 10,
    test_size: float = 0.2,
    base_seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Run each model n_runs times with different train/test splits.
    Returns:
      - per_model_class_acc: dict[name] -> (K,) array of average per-class F1
      - per_model_macro_acc: dict[name] -> macro average F1
    NOTE: We keep variable names 'acc' for minimal downstream changes,
          but the metric is now F1.
    """
    classes = np.unique(y)
    K = len(classes)

    per_model_class_sum: Dict[str, np.ndarray] = {
        name: np.zeros(K, dtype=float) for name in model_prototypes
    }
    per_model_count: Dict[str, np.ndarray] = {
        name: np.zeros(K, dtype=float) for name in model_prototypes
    }

    for run in range(n_runs):
        seed = base_seed + run
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        for name, proto in model_prototypes.items():
            model = clone(proto)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            # -------- CHANGED: per-class F1 instead of diag/row-sum --------
            per_class_f1 = f1_score(
                y_te, y_pred,
                labels=classes,
                average=None,
                zero_division=0
            )

            per_model_class_sum[name] += per_class_f1
            per_model_count[name] += 1.0

    per_model_class_acc: Dict[str, np.ndarray] = {}
    per_model_macro_acc: Dict[str, float] = {}

    for name in model_prototypes:
        counts = per_model_count[name]
        nonzero = counts > 0
        avg = np.zeros(K, dtype=float)
        avg[nonzero] = per_model_class_sum[name][nonzero] / counts[nonzero]

        per_model_class_acc[name] = avg
        per_model_macro_acc[name] = float(avg.mean())  # macro-F1

    return per_model_class_acc, per_model_macro_acc


# ---------------------------------------------------------
# SynthTree training with chosen teacher
# ---------------------------------------------------------

def train_synthtree_with_teacher(
    X: np.ndarray,
    y: np.ndarray,
    teacher_proto,
    feature_names: List[str],
    class_names: List[str],
    random_state: int = 0,
    test_size: float = 0.2,
    n_clusters: int = 10,
    n_aug_per_cluster: int = 100,
) -> Tuple[SynthTree, float, np.ndarray, pd.DataFrame]:
    """
    Train SynthTree using the chosen teacher prototype.
    Returns:
      - trained SynthTree object
      - SynthTree test accuracy (overall)
      - SynthTree per-class F1 (K,)
      - DataFrame with per-class feature importance (normalized)
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    st = SynthTree(
        n_clusters=n_clusters,
        n_aug_per_cluster=n_aug_per_cluster,
        teacher=teacher_proto,
        pruning="l_trim",
        random_state=random_state,
        n_jobs=-1,
        progress=False,
    )
    st.fit(X_tr, y_tr)
    y_pred_st = st.predict(X_te)

    st_acc = accuracy_score(y_te, y_pred_st)

    K = len(class_names)
    classes = np.arange(K, dtype=int)  # LabelEncoder -> 0..K-1
    per_class_f1 = f1_score(
        y_te, y_pred_st,
        labels=classes,
        average=None,
        zero_division=0
    )

    # Per-class FI: shape (K, p)
    if not hasattr(st, "stfi_class_normalized_"):
        raise RuntimeError("SynthTree does not expose stfi_class_normalized_.")

    fi_class = st.stfi_class_normalized_   # (K, p)
    K_fi, p = fi_class.shape

    if K_fi != K:
        if K_fi < K:
            pad = np.zeros((K - K_fi, p), dtype=float)
            fi_class = np.vstack([fi_class, pad])
        else:
            fi_class = fi_class[:K, :]

    if len(feature_names) != p:
        feature_cols = [f"f{j}" for j in range(p)]
    else:
        feature_cols = feature_names

    if len(class_names) != K:
        row_index = [f"class_{k}" for k in range(K)]
    else:
        row_index = class_names

    fi_df = pd.DataFrame(fi_class, index=row_index, columns=feature_cols)

    return st, st_acc, per_class_f1, fi_df


# ---------------------------------------------------------
# Helper: compute SynthTree depth
# ---------------------------------------------------------

def compute_tree_depth(st: SynthTree) -> int:
    if not hasattr(st, "nodes_") or not st.nodes_:
        return 0

    max_depth = 0
    for idx, n in enumerate(st.nodes_):
        d = 0
        j = idx
        while st.nodes_[j].parent is not None:
            d += 1
            j = st.nodes_[j].parent
        max_depth = max(max_depth, d)
    return max_depth


# ---------------------------------------------------------
# Auto-tuning n_clusters per dataset (minimal change)
# ---------------------------------------------------------

def tune_n_clusters_for_dataset(
    X: np.ndarray,
    y: np.ndarray,
    teacher_proto,
    feature_names: List[str],
    class_names: List[str],
    random_state: int,
    test_size: float,
    n_aug_per_cluster: int,
    candidate_n_clusters: List[int],
) -> int:
    n_total = X.shape[0]
    n_train_est = int(n_total * (1.0 - test_size))

    valid_candidates = sorted({nc for nc in candidate_n_clusters if nc <= max(1, n_train_est)})
    if not valid_candidates:
        valid_candidates = [max(1, n_train_est)]

    print(f"\n  Auto-tuning n_clusters; candidate grid (filtered): {valid_candidates}")

    best_nc = valid_candidates[0]
    best_acc = -np.inf

    for nc in valid_candidates:
        _, acc, _, _ = train_synthtree_with_teacher(
            X, y,
            teacher_proto=teacher_proto,
            feature_names=feature_names,
            class_names=class_names,
            random_state=random_state,
            test_size=test_size,
            n_clusters=nc,
            n_aug_per_cluster=n_aug_per_cluster,
        )
        print(f"    n_clusters={nc}: SynthTree acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_nc = nc

    print(f"  -> Chosen n_clusters={best_nc} (SynthTree acc={best_acc:.4f})")
    return best_nc


# ---------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="run_all_3: BB models + SynthTree + per-class FI")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with CSV files (e.g., your emotion datasets).")
    parser.add_argument("--target_col", type=str, default="emotion_label",
                        help="Name of the target column in CSVs.")
    parser.add_argument("--n_runs", type=int, default=10,
                        help="Number of train/test repetitions per black-box model.")
    parser.add_argument("--n_clusters", type=int, default=10,
                        help=("Baseline n_clusters value for SynthTree. "
                              "We will auto-tune around a small grid that includes this value."))
    parser.add_argument("--n_aug_per_cluster", type=int, default=100,
                        help="Number of augmented points per cluster for SynthTree.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} is not a directory")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"Found {len(csv_files)} CSV files in {data_dir}")

    all_bb_results = []

    base_cluster_grid = [5, 10, 15, 20]

    for csv_path in csv_files:
        dataset_name = csv_path.stem
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset_name}")
        print("=" * 80)

        X, y, feature_names, class_names, le = load_emotion_csv(
            csv_path, target_col=args.target_col
        )

        # 1) Evaluate black-box models (per-class F1 now)
        bb_models = get_black_box_models(random_state=args.seed)
        per_model_class_acc, per_model_macro_acc = evaluate_models_per_class(
            X, y, bb_models,
            n_runs=args.n_runs,
            test_size=0.2,
            base_seed=args.seed,
        )

        print("\nAverage per-class F1 (over runs):")
        for name, f1_vec in per_model_class_acc.items():
            macro = per_model_macro_acc[name]
            f1_str = ", ".join(
                f"{cls}: {f1_vec[i]:.3f}"
                for i, cls in enumerate(class_names)
            )
            print(f"  {name:>4s} | macro={macro:.4f} | {f1_str}")

            for i, cls in enumerate(class_names):
                all_bb_results.append({
                    "dataset": dataset_name,
                    "model": name,
                    "class": cls,
                    "per_class_acc": f1_vec[i],  # NOTE: stored as F1 for minimal downstream changes
                    "macro_acc": macro,          # NOTE: macro-F1
                })

        # 2) Pick best model by macro F1
        best_model_name = max(per_model_macro_acc.items(), key=lambda kv: kv[1])[0]
        best_macro = per_model_macro_acc[best_model_name]
        print(f"\nBest teacher model for dataset '{dataset_name}': {best_model_name} "
              f"(macro F1 = {best_macro:.4f})")

        best_teacher_proto = bb_models[best_model_name]

        candidate_n_clusters = list(base_cluster_grid)
        if args.n_clusters not in candidate_n_clusters:
            candidate_n_clusters.append(args.n_clusters)
        candidate_n_clusters = sorted(set(candidate_n_clusters))

        best_n_clusters = tune_n_clusters_for_dataset(
            X, y,
            teacher_proto=best_teacher_proto,
            feature_names=feature_names,
            class_names=class_names,
            random_state=args.seed,
            test_size=0.2,
            n_aug_per_cluster=args.n_aug_per_cluster,
            candidate_n_clusters=candidate_n_clusters,
        )

        # 3) Train SynthTree with chosen teacher and chosen n_clusters
        st, synth_acc, synth_per_class_f1, fi_df = train_synthtree_with_teacher(
            X, y,
            teacher_proto=best_teacher_proto,
            feature_names=feature_names,
            class_names=class_names,
            random_state=args.seed,
            test_size=0.2,
            n_clusters=best_n_clusters,
            n_aug_per_cluster=args.n_aug_per_cluster,
        )

        tree_depth = compute_tree_depth(st)
        print(f"SynthTree depth on '{dataset_name}' (n_clusters={best_n_clusters}): {tree_depth}")

        print(f"\nSynthTree test accuracy on '{dataset_name}' "
              f"(n_clusters={best_n_clusters}): {synth_acc:.4f}")

        synth_macro = float(synth_per_class_f1.mean())
        f1_str = ", ".join(
            f"{cls}: {synth_per_class_f1[i]:.3f}"
            for i, cls in enumerate(class_names)
        )
        print(f"SynthTree per-class F1 (macro={synth_macro:.4f}): {f1_str}")

        for i, cls in enumerate(class_names):
            all_bb_results.append({
                "dataset": dataset_name,
                "model": "SynthTree",
                "class": cls,
                "per_class_acc": synth_per_class_f1[i],  # NOTE: stored as F1
                "macro_acc": synth_macro,                # NOTE: macro-F1
            })

        print("\nPer-class SynthTree feature importance (rows = classes, cols = features):")
        for cls in fi_df.index:
            row = fi_df.loc[cls]
            top_feats = row.sort_values(ascending=False).head(5)
            top_str = ", ".join(f"{f}:{v:.3f}" for f, v in top_feats.items())
            print(f"  {cls}: {top_str}")

        output_dir = data_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        out_fi_path = output_dir / f"fi_synthtree_{dataset_name}_{best_model_name}.csv"
        fi_df.to_csv(out_fi_path, index=True)
        print(f"\nSaved per-class SynthTree FI to: {out_fi_path}")

        try:
            dot_source = st.visualize()
            dot_path = output_dir / f"synthtree_{dataset_name}_{best_model_name}.dot"
            with open(dot_path, "w") as f:
                f.write(dot_source)
            print(f"Saved SynthTree DOT plot to: {dot_path}")
        except Exception as e:
            print(f"[WARN] Could not generate DOT for SynthTree: {e}")

        print(f"Chosen n_clusters for dataset '{dataset_name}': {best_n_clusters}")

    if all_bb_results:
        bb_df = pd.DataFrame(all_bb_results)
        out_bb_path = data_dir / "bb_per_class_accuracy_summary_small_tree.csv"
        bb_df.to_csv(out_bb_path, index=False)
        print(f"\nSaved per-class summary (BB + SynthTree) to: {out_bb_path}")
        print("NOTE: per_class_acc/macro_acc columns now store F1, not accuracy.")


if __name__ == "__main__":
    main()