from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

# ----------------------------
# Progress Bar
# ----------------------------
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):  # fallback no-op
        return x

# ----------------------------
# Constants-like models
# ----------------------------

class ConstantBinaryClassifier(BaseEstimator):
    """
    Sklearn-like binary classifier that always predicts the sole observed class.
    Provides predict & predict_proba; exposes classes_, coef_, intercept_.
    """
    def __init__(self):
        self.constant_label_ = None
        self.classes_ = np.array([0, 1])
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = check_array(X, dtype=float)
        y = np.asarray(y)
        uniq = np.unique(y)
        if len(uniq) == 0:
            raise ValueError("Empty y in ConstantBinaryClassifier.")
        if not set(uniq).issubset({0, 1}):
            raise ValueError("ConstantBinaryClassifier expects binary labels {0,1}.")
        if len(uniq) == 1:
            self.constant_label_ = int(uniq[0])
        else:
            self.constant_label_ = int(np.bincount(y.astype(int)).argmax())
        self.coef_ = np.zeros((1, X.shape[1]), dtype=float)
        self.intercept_ = np.array([0.0], dtype=float)
        return self

    def predict(self, X):
        check_is_fitted(self, "constant_label_")
        X = check_array(X, dtype=float)
        return np.full(X.shape[0], self.constant_label_, dtype=int)

    def predict_proba(self, X):
        check_is_fitted(self, "constant_label_")
        X = check_array(X, dtype=float)
        if self.constant_label_ == 1:
            return np.column_stack([np.zeros(X.shape[0]), np.ones(X.shape[0])])
        else:
            return np.column_stack([np.ones(X.shape[0]), np.zeros(X.shape[0])])


class ConstantMulticlassClassifier(BaseEstimator):
    """Always predicts the majority class it saw; exposes classes_ and predict_proba(K)."""
    def __init__(self, n_classes=None):
        self.n_classes_ = n_classes
        self.constant_label_ = None
        self.classes_ = None
        self.coef_ = None  # keep attribute for compatibility

    def fit(self, X, y):
        X = check_array(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if y.size == 0:
            raise ValueError("Empty y in ConstantMulticlassClassifier.")
        counts = np.bincount(y)
        self.constant_label_ = int(np.argmax(counts))
        if self.n_classes_ is None:
            self.n_classes_ = int(np.max(y)) + 1
        self.classes_ = np.arange(self.n_classes_, dtype=int)
        self.coef_ = np.zeros((1, X.shape[1]), dtype=float)
        return self

    def predict(self, X):
        X = check_array(X, dtype=float)
        return np.full(X.shape[0], self.constant_label_, dtype=int)

    def predict_proba(self, X):
        X = check_array(X, dtype=float)
        P = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        P[:, self.constant_label_] = 1.0
        return P


class ConstantRegressor(BaseEstimator):
    """Sklearn-like regressor that predicts the mean of y."""
    def __init__(self):
        self.ybar_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = check_array(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("Empty y in ConstantRegressor.")
        self.ybar_ = float(np.mean(y))
        self.coef_ = np.zeros((1, X.shape[1]), dtype=float)
        self.intercept_ = np.array([self.ybar_], dtype=float)
        return self

    def predict(self, X):
        check_is_fitted(self, "ybar_")
        X = check_array(X, dtype=float)
        return np.full(X.shape[0], self.ybar_, dtype=float)

# ----------------------------
# Internal node/leaf structure
# ----------------------------

@dataclass
class _Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional[int] = None
    right: Optional[int] = None
    model_indices: Optional[np.ndarray] = None  # which cell-models route here
    is_leaf: bool = False
    leaf_model: Optional[Any] = None
    leaf_id: Optional[int] = None
    # For STFI / structure introspection
    parent: Optional[int] = None
    gain: Optional[float] = None  # impurity reduction used at this split (if internal)

# ----------------------------
# SynthTree Estimator
# ----------------------------

class SynthTree(BaseEstimator):
    """
    SynthTree estimator (sklearn-style).

    Steps:
      1) K-means cells; Gaussian augmentation per cell; labels from teacher;
         fit sparse local linear models g_j per cell.
      2) Grow a binary tree on {g_j} using single-feature thresholds; impurity =
         average pairwise mutual prediction disparity; split on positive CART-style gain.
      3) Refit sparse linear/logistic models in each leaf on original+aug points in that region.
      4) Prune (optional): L-trim or cost-complexity.
      5) STFI: feature importance from path gains + leaf-model coefficients, weighted by leaf support.
         Also compute per-class STFI for classification tasks.
    """

    def __init__(
        self,
        n_clusters: int = 100,
        n_aug_per_cluster: int = 100,
        teacher: Union[BaseEstimator, List[BaseEstimator], None] = None,
        pruning: Optional[str] = "l_trim",
        random_state: Optional[int] = None,
        verbose: int = 0,
        distance_metric: str = "accuracy",
        n_jobs: int = 1,
        progress: bool = False
    ):
        self.n_clusters = n_clusters
        self.n_aug_per_cluster = n_aug_per_cluster
        self.teacher = teacher
        self.pruning = pruning
        self.random_state = random_state
        self.verbose = verbose
        self.distance_metric = distance_metric
        self.n_jobs = n_jobs
        self.progress = progress

    # ----------------------------
    # Public API
    # ----------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SynthTree":
        X, y = check_X_y(X, y, accept_sparse=False, dtype=float, multi_output=False)
        self.n_samples_, self.n_features_ = X.shape

        # Task + global classes
        y_vals = np.unique(y)
        n_classes = len(y_vals)
        if n_classes <= 2 and set(y_vals).issubset({0.0, 1.0}):
            self.task_ = "binary"
        elif n_classes >= 3 and np.issubdtype(y.dtype, np.integer):
            self.task_ = "multiclass"
        else:
            self.task_ = "regression"
        self.n_classes_ = n_classes if self.task_ in ("binary", "multiclass") else None
        if self.task_ in ("binary", "multiclass"):
            self.classes_ = np.asarray(sorted(y_vals.astype(int)))

        # Cache training set for pruning + STFI
        self._train_X_ = X
        self._train_y_ = y

        rng = np.random.RandomState(self.random_state)

        # 1) KMeans cells
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, n_init=10,
            random_state=self.random_state
        )
        labels = self.kmeans_.fit_predict(X)
        centroids = self.kmeans_.cluster_centers_
        cells = [np.where(labels == j)[0] for j in range(self.n_clusters)]

        # Fit teacher(s)
        teachers = self._prepare_teachers(X, y)
        teacher_global_scores = self._teacher_global_scores(teachers, X, y)

        # Per-cell augmentation + local models g_j
        from joblib import Parallel, delayed

        def _fit_one_cell(j: int, idx: np.ndarray):
            if len(idx) == 0:
                return j, None, None
            Xj, yj = X[idx], y[idx]
            xbar = centroids[j]

            # diag variances; robust for tiny clusters
            n_j = Xj.shape[0]
            if n_j >= 2:
                varj = Xj.var(axis=0, ddof=1)
            else:
                varj = Xj.var(axis=0, ddof=0)
            varj = np.where(np.isfinite(varj), varj, 0.0)
            varj = np.maximum(varj, 1e-6)

            # choose teacher per cell if list provided
            if isinstance(teachers, list):
                scores = []
                for t in teachers:
                    y_hat = t.predict(Xj)
                    s = accuracy_score(yj, y_hat) if self.task_ in ("binary", "multiclass") \
                        else -mean_squared_error(yj, y_hat)
                    scores.append(s)
                best = int(np.argmax(scores))
                ties = np.where(np.isclose(scores, scores[best]))[0]
                if len(ties) > 1:
                    best = ties[np.argmax([teacher_global_scores[k] for k in ties])]
                teacher_j = teachers[best]
            else:
                teacher_j = teachers

            # deterministic per-cell RNG for reproducibility
            rng_cell = np.random.RandomState(
                None if self.random_state is None else self.random_state + j
            )

            # augment
            X_aug = rng_cell.multivariate_normal(
                mean=xbar, cov=np.diag(varj), size=self.n_aug_per_cluster
            )
            y_aug = teacher_j.predict(X_aug)

            # combine & fit sparse local model g_j
            Xj_full = np.vstack([Xj, X_aug])
            yj_full = np.concatenate([yj, y_aug])
            gm = self._fit_sparse_model(Xj_full, yj_full)
            return j, (Xj_full, yj_full, xbar), gm

        if self.n_jobs is None or self.n_jobs == 1:
            it = enumerate(cells)
            if self.progress:
                it = tqdm(it, total=len(cells), desc="Fitting cells")
            results = [_fit_one_cell(j, idx) for j, idx in it]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_one_cell)(j, idx) for j, idx in enumerate(cells)
            )

        # reassemble in original order
        cell_data = [None] * self.n_clusters
        cell_models = [None] * self.n_clusters
        for j, data_j, model_j in results:
            cell_data[j] = data_j
            cell_models[j] = model_j

        # keep non-empty
        self.valid_cells_ = np.array(
            [j for j, d in enumerate(cell_data) if d is not None], dtype=int
        )
        if len(self.valid_cells_) == 0:
            raise RuntimeError("No non-empty k-means cells were formed.")

        self.cell_data_ = [cell_data[j] for j in self.valid_cells_]
        self.cell_models_ = [cell_models[j] for j in self.valid_cells_]
        self.cell_centroids_ = np.vstack(
            [centroids[j] for j in self.valid_cells_]
        )

        self._precompute_pairwise_distances()

        # 2) Grow tree on {g_j}
        self.nodes_: List[_Node] = []
        root = _Node(model_indices=np.arange(len(self.valid_cells_)), parent=None)
        self.nodes_.append(root)
        self._grow_node(0)

        # 3) Refit leaf models on final partitions (original + augmented)
        self._refit_leaves(X, y)

        # 4) Prune
        if self.pruning in ("l_trim", "cc_prune"):
            self._prune(self._train_X_, self._train_y_)

        # Assign final leaf ids
        self._assign_leaf_ids()

        # 5) STFI (global) and per-class STFI (for classification)
        self.stfi_raw_, self.stfi_normalized_ = self._compute_stfi()
        self.feature_importances_ = self.stfi_normalized_

        if self.task_ in ("binary", "multiclass"):
            (self.stfi_class_raw_,
             self.stfi_class_normalized_) = self._compute_stfi_per_class()

        return self

    def predict(self, X):
        check_is_fitted(self, "nodes_")
        X = check_array(X, dtype=float)
        if self.task_ in ("binary", "multiclass"):
            P = self.predict_proba(X)
            if self.task_ == "multiclass":
                return np.argmax(P, axis=1).astype(int)
            else:
                return (P[:, 1] >= 0.5).astype(int)
        out = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            node_idx = self._traverse(X[i])
            out[i] = self.nodes_[node_idx].leaf_model.predict(X[i][None, :])[0]
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "nodes_")
        X = check_array(X, dtype=float)
        if self.task_ not in ("binary", "multiclass"):
            raise AttributeError("predict_proba is only available for classification.")

        n = X.shape[0]
        K = self.n_classes_
        out = np.zeros((n, K), dtype=float)

        for i in range(n):
            node_idx = self._traverse(X[i])
            m = self.nodes_[node_idx].leaf_model
            Pi = m.predict_proba(X[i][None, :])  # shape (1, K_leaf)

            # map leaf classes -> global classes
            if hasattr(m, "classes_") and m.classes_ is not None:
                cls = np.asarray(m.classes_, dtype=int)
            else:
                cls = np.arange(Pi.shape[1], dtype=int)
            for c_local, c_value in enumerate(cls):
                g_idx = int(np.where(self.classes_ == c_value)[0][0])
                out[i, g_idx] = Pi[0, c_local]

            s = out[i].sum()
            if s > 0:
                out[i] /= s
            else:
                # backstop for degenerate cases
                yhat = int(m.predict(X[i][None, :])[0])
                g_idx = int(np.where(self.classes_ == yhat)[0][0])
                out[i, g_idx] = 1.0
        return out

    # ----------------------------
    # Introspection / Visualization
    # ----------------------------

    def get_leaf_models(self) -> Dict[int, Any]:
        check_is_fitted(self, "nodes_")
        return {
            n.leaf_id: n.leaf_model
            for n in self.nodes_
            if n.is_leaf and n.leaf_model is not None
        }

    def visualize(self) -> str:
        check_is_fitted(self, "nodes_")
        reachable = set(self._reachable_node_indices())
        try:
            from graphviz import Digraph
            dot = Digraph("SynthTree")
            for idx, node in enumerate(self.nodes_):
                if idx not in reachable:
                    continue
                if node.is_leaf:
                    label = f"Leaf {node.leaf_id}\\nmodel={type(node.leaf_model).__name__}"
                    dot.node(str(idx), label=label, shape="box")
                else:
                    label = f"X[{node.feature}] <= {node.threshold:.6g}"
                    dot.node(str(idx), label=label)
            for idx, node in enumerate(self.nodes_):
                if idx not in reachable or node.is_leaf:
                    continue
                dot.edge(str(idx), str(node.left), label="Yes")
                dot.edge(str(idx), str(node.right), label="No")
            return dot.source
        except Exception:
            lines = []

            def rec(i, ind=""):
                n = self.nodes_[i]
                if n.is_leaf:
                    lines.append(f"{ind}Leaf {n.leaf_id}: {type(n.leaf_model).__name__}")
                else:
                    lines.append(f"{ind}if X[{n.feature}] <= {n.threshold:.6g}:")
                    rec(n.left, ind + "  ")
                    lines.append(f"{ind}else:")
                    rec(n.right, ind + "  ")

            if self.nodes_:
                rec(0)
            return "\n".join(lines)

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _fit_sparse_model(self, X, y):
        if self.task_ == "binary":
            uniq = np.unique(y)
            if len(uniq) == 1:
                return ConstantBinaryClassifier().fit(X, y.astype(int))
            return LogisticRegression(
                penalty="l1",
                solver="liblinear",
                random_state=self.random_state
            ).fit(X, y.astype(int))
        elif self.task_ == "multiclass":
            uniq = np.unique(y)
            if len(uniq) == 1:
                return ConstantMulticlassClassifier(
                    n_classes=self.n_classes_
                ).fit(X, y.astype(int))
            return LogisticRegression(
                penalty="l1",
                solver="saga",
                random_state=self.random_state,
                max_iter=2000
            ).fit(X, y.astype(int))
        else:
            return Lasso(alpha=1.0, random_state=self.random_state).fit(
                X, y.astype(float)
            )

    def _prepare_teachers(self, X, y):
        if self.teacher is None:
            raise ValueError("A black-box teacher (or list of teachers) must be provided.")
        from sklearn.base import clone

        def clone_and_fit(est):
            e = clone(est)
            if hasattr(e, "random_state"):
                e.random_state = self.random_state
            e.fit(X, y)
            return e

        if isinstance(self.teacher, list):
            return [clone_and_fit(t) for t in self.teacher]
        return clone_and_fit(self.teacher)

    def _teacher_global_scores(self, teachers, X, y):
        if not isinstance(teachers, list):
            return None
        scores = []
        for t in teachers:
            y_hat = t.predict(X)
            if self.task_ in ("binary", "multiclass"):
                s = accuracy_score(y, y_hat)
            else:
                s = -mean_squared_error(y, y_hat)
            scores.append(s)
        return scores

    def _grow_node(self, node_idx: int):
        node = self.nodes_[node_idx]
        idxs = node.model_indices
        if idxs is None or len(idxs) <= 1:
            node.is_leaf = True
            return

        Xbar = self.cell_centroids_[idxs]
        best_gain = -np.inf
        best_split = None
        D_parent = self._node_impurity(idxs)

        for d in range(self.n_features_):
            vals = Xbar[:, d]
            uniq = np.unique(vals)
            if len(uniq) <= 1:
                continue
            uniq.sort()
            mids = (uniq[:-1] + uniq[1:]) / 2.0
            for thr in mids:
                left_mask = vals <= thr
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                left_idxs = idxs[left_mask]
                right_idxs = idxs[right_mask]
                pL = len(left_idxs) / len(idxs)
                pR = 1.0 - pL
                D_L = self._node_impurity(left_idxs)
                D_R = self._node_impurity(right_idxs)

                if not (np.isfinite(D_parent) and np.isfinite(D_L) and np.isfinite(D_R)):
                    continue
                gain = D_parent - pL * D_L - pR * D_R
                if not np.isfinite(gain):
                    continue

                if gain > best_gain:
                    best_gain = gain
                    best_split = (d, float(thr), left_idxs, right_idxs)

        if best_split is None or best_gain <= 1e-12:
            node.is_leaf = True
            return

        d, thr, left_idxs, right_idxs = best_split
        node.feature = d
        node.threshold = thr
        node.is_leaf = False
        node.gain = float(best_gain)

        left_node = _Node(model_indices=left_idxs, parent=node_idx)
        right_node = _Node(model_indices=right_idxs, parent=node_idx)
        node.left = len(self.nodes_)
        self.nodes_.append(left_node)
        node.right = len(self.nodes_)
        self.nodes_.append(right_node)

        self._grow_node(node.left)
        self._grow_node(node.right)

    def _node_impurity(self, model_indices: np.ndarray) -> float:
        if len(model_indices) <= 1:
            return 0.0

        if hasattr(self, "_pairwise_dist_") and self._pairwise_dist_ is not None:
            idxs = np.asarray(model_indices, dtype=int)
            k = idxs.size
            if k < 2:
                return 0.0
            subm = self._pairwise_dist_[np.ix_(idxs, idxs)]
            iu = np.triu_indices(k, 1)
            return float(subm[iu].mean()) if iu[0].size > 0 else 0.0

        # (fallback not usually used after precompute)
        acc = 0.0
        cnt = 0
        for i in range(len(model_indices)):
            j1 = model_indices[i]
            g1 = self.cell_models_[j1]
            X1, _, _ = self.cell_data_[j1]
            for k in range(i + 1, len(model_indices)):
                j2 = model_indices[k]
                g2 = self.cell_models_[j2]
                X2, _, _ = self.cell_data_[j2]
                d = self._model_distance(g1, X1, g2, X2)
                acc += d
                cnt += 1
        return acc / cnt

    def _model_distance(self, g1, X1, g2, X2) -> float:
        Xcat = np.vstack([X1, X2])
        if self.task_ in ("binary", "multiclass"):
            P1 = g1.predict_proba(Xcat)
            P2 = g2.predict_proba(Xcat)

            def to_global_argmax(P, m):
                if (hasattr(m, "classes_") and m.classes_ is not None
                        and P.shape[1] != self.n_classes_):
                    G = np.zeros((P.shape[0], self.n_classes_), dtype=float)
                    for c_local, c_value in enumerate(np.asarray(m.classes_, dtype=int)):
                        g_idx = int(np.where(self.classes_ == c_value)[0][0])
                        G[:, g_idx] = P[:, c_local]
                    P = G
                return np.argmax(P, axis=1)

            y1 = to_global_argmax(P1, g1)
            y2 = to_global_argmax(P2, g2)
            acc = np.mean(y1 == y2)
            return 1.0 - float(acc)
        else:
            y1 = g1.predict(Xcat)
            y2 = g2.predict(Xcat)
            return float(np.mean((y1 - y2) ** 2))

    def _precompute_pairwise_distances(self):
        """Compute and cache symmetric distance matrix between all cell models."""
        m = len(self.cell_models_)
        self._pairwise_dist_ = np.zeros((m, m), dtype=float)
        if m <= 1:
            return

        try:
            from joblib import Parallel, delayed
            use_joblib = True
        except Exception:
            use_joblib = False

        def _pair(i, j):
            gi, Xi = self.cell_models_[i], self.cell_data_[i][0]
            gj, Xj = self.cell_models_[j], self.cell_data_[j][0]
            return i, j, float(self._model_distance(gi, Xi, gj, Xj))

        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]

        if use_joblib and (self.n_jobs is None or self.n_jobs != 1):
            triples = Parallel(n_jobs=self.n_jobs)(
                delayed(_pair)(i, j) for (i, j) in pairs
            )
        else:
            it = pairs
            if getattr(self, "progress", False):
                it = tqdm(it, total=len(pairs), desc="Pairwise distances")
            triples = (_pair(i, j) for (i, j) in it)

        for i, j, d in triples:
            self._pairwise_dist_[i, j] = d
            self._pairwise_dist_[j, i] = d

    def _traverse(self, x: np.ndarray) -> int:
        idx = 0
        while True:
            n = self.nodes_[idx]
            if n.is_leaf:
                return idx
            idx = n.left if x[n.feature] <= n.threshold else n.right

    def _collect_leaf_region_indices(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        leaf_to_idx = {}
        for i in range(X.shape[0]):
            ln = self._traverse(X[i])
            leaf_to_idx.setdefault(ln, []).append(i)
        return {k: np.array(v, dtype=int) for k, v in leaf_to_idx.items()}

    def _refit_leaves(self, X: np.ndarray, y: np.ndarray):
        # concatenate all cell data once
        X_aug_all = np.vstack([d[0] for d in self.cell_data_])
        y_aug_all = np.concatenate([d[1] for d in self.cell_data_])

        Xa = np.vstack([X, X_aug_all])
        ya = np.concatenate([y, y_aug_all])

        leaf_assign = self._collect_leaf_region_indices(Xa)

        items = list(leaf_assign.items())
        if getattr(self, "progress", False) and (self.n_jobs is None or self.n_jobs == 1):
            items = tqdm(items, total=len(items), desc="Refitting leaves")

        try:
            from joblib import Parallel, delayed
            if self.n_jobs is None or self.n_jobs == 1:
                fitted = [
                    (node_idx, self._fit_sparse_model(Xa[idxs], ya[idxs]))
                    for node_idx, idxs in items
                ]
            else:
                def _fit_leaf(ni):
                    node_idx, idxs = ni
                    return node_idx, self._fit_sparse_model(Xa[idxs], ya[idxs])
                fitted = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_leaf)(ni) for ni in leaf_assign.items()
                )
        except Exception:
            fitted = [
                (node_idx, self._fit_sparse_model(Xa[idxs], ya[idxs]))
                for node_idx, idxs in items
            ]

        for node_idx, lm in fitted:
            n = self.nodes_[node_idx]
            n.is_leaf = True
            n.leaf_model = lm

        # fallback for any degenerate leaf
        for n in self.nodes_:
            if n.is_leaf and n.leaf_model is None:
                Xb, yb = [], []
                if n.model_indices is not None:
                    for j in n.model_indices.tolist():
                        Xj, yj, _ = self.cell_data_[j]
                        Xb.append(Xj)
                        yb.append(yj)
                if len(Xb) > 0:
                    Xb = np.vstack(Xb)
                    yb = np.concatenate(yb)
                    n.leaf_model = self._fit_sparse_model(Xb, yb)

    def _assign_leaf_ids(self):
        lid = 0
        reachable = set(self._reachable_node_indices())
        for idx, n in enumerate(self.nodes_):
            if idx in reachable and n.is_leaf:
                n.leaf_id = lid
                lid += 1
            else:
                n.leaf_id = None

    def _reachable_node_indices(self) -> List[int]:
        if not hasattr(self, "nodes_") or not self.nodes_:
            return []
        reachable = []
        stack = [0]
        seen = set()
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            reachable.append(i)
            n = self.nodes_[i]
            if not n.is_leaf:
                if n.left is not None:
                    stack.append(n.left)
                if n.right is not None:
                    stack.append(n.right)
        return reachable

    # ----------------------------
    # Pruning
    # ----------------------------

    def _collect_nodes_by_depth(self) -> Dict[int, List[int]]:
        depth_map: Dict[int, List[int]] = {}

        def walk(i: int, d: int):
            n = self.nodes_[i]
            depth_map.setdefault(d, []).append(i)
            if not n.is_leaf:
                walk(n.left, d + 1)
                walk(n.right, d + 1)

        walk(0, 0)
        return depth_map

    def _clone_with_pruned_to_depth(self, max_depth: int) -> "SynthTree":
        clone = self._structural_clone()

        def prune(i: int, d: int):
            n = clone.nodes_[i]
            if d >= max_depth:
                n.is_leaf = True
                n.left = n.right = None
                n.feature = n.threshold = None
                n.gain = None
                return
            if not n.is_leaf and n.left is not None and n.right is not None:
                prune(n.left, d + 1)
                prune(n.right, d + 1)

        prune(0, 0)
        clone._refit_leaves(self._train_X_, self._train_y_)
        clone._assign_leaf_ids()
        return clone

    def _structural_clone(self) -> "SynthTree":
        clone = SynthTree(
            n_clusters=self.n_clusters,
            n_aug_per_cluster=self.n_aug_per_cluster,
            teacher=self.teacher,
            pruning=self.pruning,
            random_state=self.random_state,
            verbose=self.verbose,
            distance_metric=self.distance_metric,
            n_jobs=self.n_jobs,
        )
        clone.task_ = self.task_
        clone.n_features_ = self.n_features_
        if hasattr(self, "n_classes_"):
            clone.n_classes_ = self.n_classes_
        if hasattr(self, "classes_"):
            clone.classes_ = self.classes_
        clone.cell_data_ = self.cell_data_
        clone.cell_models_ = self.cell_models_
        clone.cell_centroids_ = self.cell_centroids_
        clone.valid_cells_ = self.valid_cells_
        clone.kmeans_ = self.kmeans_
        clone.n_samples_ = self.n_samples_
        clone._train_X_ = self._train_X_
        clone._train_y_ = self._train_y_
        clone.nodes_ = []
        for n in self.nodes_:
            clone.nodes_.append(_Node(
                feature=n.feature, threshold=n.threshold,
                left=n.left, right=n.right,
                model_indices=None if n.model_indices is None else n.model_indices.copy(),
                is_leaf=n.is_leaf, leaf_model=None, leaf_id=None,
                parent=n.parent, gain=n.gain
            ))
        if hasattr(self, "_pairwise_dist_"):
            clone._pairwise_dist_ = self._pairwise_dist_
        return clone

    def _cost_complexity_sequence(self) -> Tuple[List[float], List["SynthTree"]]:
        Tk = self._structural_clone()
        Tk._refit_leaves(self._train_X_, self._train_y_)
        Tk._assign_leaf_ids()

        alphas = [0.0]
        subtrees = [Tk]
        prev_alpha = 0.0
        while True:
            hk_vals = []
            internal_nodes = []
            for i, n in enumerate(Tk.nodes_):
                if n.is_leaf:
                    continue
                hk_vals.append(self._hk_for_node(Tk, i))
                internal_nodes.append(i)
            if len(hk_vals) == 0:
                break
            hk_arr = np.array(hk_vals, dtype=float)
            mask = hk_arr >= prev_alpha - 1e-12
            if not mask.any():
                break
            alpha_next = float(np.min(hk_arr[mask]))

            Tk_next = Tk._structural_clone()
            for node_idx, hk in zip(internal_nodes, hk_arr):
                if hk <= alpha_next + 1e-12:
                    self._collapse_subtree(Tk_next, node_idx)
            Tk_next._refit_leaves(self._train_X_, self._train_y_)
            Tk_next._assign_leaf_ids()

            alphas.append(alpha_next)
            subtrees.append(Tk_next)
            Tk = Tk_next
            prev_alpha = alpha_next
            if sum(n.is_leaf for n in Tk.nodes_) == 1 and len(Tk.nodes_) == 1:
                break
        return alphas, subtrees

    def _hk_for_node(self, T: "SynthTree", node_idx: int) -> float:
        subtree_nodes = self._subtree_nodes(T, node_idx)
        leaf_nodes = [i for i in subtree_nodes if T.nodes_[i].is_leaf]
        Rt = self._resub_error_single_node(T, node_idx)
        RTt = sum(self._resub_error_single_node(T, i) for i in leaf_nodes)
        denom = max(1, len(leaf_nodes) - 1)
        return (Rt - RTt) / denom

    def _collapse_subtree(self, T: "SynthTree", node_idx: int):
        n = T.nodes_[node_idx]
        n.is_leaf = True
        n.left = n.right = None
        n.feature = n.threshold = None
        n.gain = None

    def _subtree_nodes(self, T: "SynthTree", node_idx: int) -> List[int]:
        out = []
        stack = [node_idx]
        while stack:
            i = stack.pop()
            out.append(i)
            n = T.nodes_[i]
            if not n.is_leaf:
                stack.extend([n.left, n.right])
        return out

    def _fit_temp_model_for_indices(self, X: np.ndarray, y: np.ndarray):
        """Fit node-level model m_t(x) for pruning."""
        return self._fit_sparse_model(X, y)

    def _resub_error_single_node(self, T: "SynthTree", node_idx: int) -> float:
        idxs = []
        for i in range(self._train_X_.shape[0]):
            if T._traverse(self._train_X_[i]) == node_idx:
                idxs.append(i)
        if len(idxs) == 0:
            return 0.0
        idxs = np.array(idxs, dtype=int)
        Xt, yt = self._train_X_[idxs], self._train_y_[idxs]
        model_t = self._fit_temp_model_for_indices(Xt, yt)

        if self.task_ in ("binary", "multiclass"):
            yhat = model_t.predict(Xt).astype(int)
            return float(np.sum(yhat != yt))
        else:
            yhat = model_t.predict(Xt)
            return float(np.sum((yt - yhat) ** 2))

    def _prune(self, X: np.ndarray, y: np.ndarray):
        if self.pruning == "l_trim":
            depth_map = self._collect_nodes_by_depth()
            max_depth = max(depth_map.keys())
            subtrees = [
                self._clone_with_pruned_to_depth(d)
                for d in range(1, max_depth + 1)
            ]
            best_idx = self._cv_select(subtrees, X, y)
            chosen = subtrees[best_idx]
            self.nodes_ = chosen.nodes_
            self._assign_leaf_ids()
        elif self.pruning == "cc_prune":
            alphas, subtrees = self._cost_complexity_sequence()
            if len(subtrees) == 1:
                return
            best_idx = self._cv_select(subtrees, X, y)
            chosen = subtrees[best_idx]
            self.nodes_ = chosen.nodes_
            self._assign_leaf_ids()
        else:
            return

    def _cv_select(self, subtrees: List["SynthTree"], X: np.ndarray, y: np.ndarray) -> int:
        if self.task_ in ("binary", "multiclass"):
            if len(np.unique(y)) == 1:
                folds = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
            else:
                folds = StratifiedKFold(
                    n_splits=10, shuffle=True, random_state=self.random_state
                )
        else:
            folds = KFold(n_splits=10, shuffle=True, random_state=self.random_state)

        scores = np.zeros((len(subtrees), 10), dtype=float)

        try:
            from joblib import Parallel, delayed
            use_joblib = True
        except Exception:
            use_joblib = False

        for f, (tr, va) in enumerate(
            folds.split(X, y if self.task_ in ("binary", "multiclass") else None)
        ):
            Xtr, ytr = X[tr], y[tr]
            Xva, yva = X[va], y[va]

            def _eval_one(i: int) -> Tuple[int, float]:
                T = subtrees[i]
                T_fold = T._structural_clone()
                T_fold._refit_leaves(Xtr, ytr)
                if self.task_ == "binary":
                    try:
                        p = T_fold.predict_proba(Xva)[:, 1]
                        s = roc_auc_score(yva, p)
                    except Exception:
                        yhat = T_fold.predict(Xva)
                        s = accuracy_score(yva, yhat)
                    return i, float(s)
                elif self.task_ == "multiclass":
                    yhat = T_fold.predict(Xva)
                    s = accuracy_score(yva, yhat)
                    return i, float(s)
                else:
                    yhat = T_fold.predict(Xva)
                    rmse = np.sqrt(mean_squared_error(yva, yhat))
                    return i, float(-rmse)

            if use_joblib and (self.n_jobs is None or self.n_jobs != 1):
                fold_results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_eval_one)(i) for i in range(len(subtrees))
                )
            else:
                fold_results = (_eval_one(i) for i in range(len(subtrees)))

            for i, s in fold_results:
                scores[i, f] = s

        return int(np.argmax(scores.mean(axis=1)))

    # ----------------------------
    # STFI (global + per-class)
    # ----------------------------

    def _compute_leaf_support_weights(self) -> Tuple[Dict[int, float], int]:
        """Compute w_ell based on counts from the same (X + aug) dataset used for leaf refits."""
        X_aug_all = np.vstack([d[0] for d in self.cell_data_])
        y_aug_all = np.concatenate([d[1] for d in self.cell_data_])
        Xa = np.vstack([self._train_X_, X_aug_all])
        leaf_assign = self._collect_leaf_region_indices(Xa)
        total = sum(len(v) for v in leaf_assign.values())
        weights = {
            node_idx: (len(idxs) / total if total > 0 else 0.0)
            for node_idx, idxs in leaf_assign.items()
        }
        return weights, Xa.shape[0]

    def _path_feature_vector_for_leaf(self, leaf_idx: int) -> np.ndarray:
        """Sum split gains along the path for each feature."""
        vec = np.zeros(self.n_features_, dtype=float)
        i = leaf_idx
        while True:
            n = self.nodes_[i]
            if n.parent is None:
                break
            p = self.nodes_[n.parent]
            if (p.feature is not None) and (p.gain is not None) and (p.gain > 0):
                vec[p.feature] += p.gain
            i = n.parent
        return vec

    def _leaf_coef_magnitude(self, model) -> np.ndarray:
        """Return |coef| per feature from the leaf model (multiclass: L1 across classes)."""
        v = np.zeros(self.n_features_, dtype=float)
        if hasattr(model, "coef_") and model.coef_ is not None:
            coef = model.coef_
            if coef.ndim == 1:
                v[:coef.shape[0]] = np.abs(coef)
            elif coef.ndim == 2:
                v[:coef.shape[1]] = np.sum(np.abs(coef), axis=0)
        return v

    def _compute_stfi(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute raw and normalized STFI (global)."""
        w_leaf, _ = self._compute_leaf_support_weights()
        stfi = np.zeros(self.n_features_, dtype=float)

        reachable = set(self._reachable_node_indices())
        for idx, n in enumerate(self.nodes_):
            if (idx not in reachable) or (not n.is_leaf) or (n.leaf_model is None):
                continue
            v_path = self._path_feature_vector_for_leaf(idx)
            v_coef = self._leaf_coef_magnitude(n.leaf_model)
            v_leaf = v_path + v_coef
            w = w_leaf.get(idx, 0.0)
            stfi += w * v_leaf

        denom = np.sum(stfi)
        stfi_norm = stfi / denom if denom > 0 else stfi
        return stfi, stfi_norm

    # ---------- NEW: per-class leaf coefficients and per-class STFI ----------

    def _leaf_coef_magnitude_per_class(self, model) -> np.ndarray:
        """
        Return |coef| per feature for each global class.
        Shape: (K, p), where K = self.n_classes_.
        Robust to:
          - binary LogisticRegression with coef_.shape == (1, p) and classes_ = [0,1]
          - Constant* classifiers with 1 row in coef_
        """
        if self.task_ not in ("binary", "multiclass"):
            return np.zeros((0, self.n_features_), dtype=float)

        K = self.n_classes_
        V = np.zeros((K, self.n_features_), dtype=float)

        if not hasattr(model, "coef_") or model.coef_ is None:
            return V

        coef = model.coef_

        # local classes from the model
        if hasattr(model, "classes_") and model.classes_ is not None:
            local_classes = np.asarray(model.classes_, dtype=int)
        else:
            # fallback: assume 0..(rows-1)
            if coef.ndim == 1:
                local_classes = np.arange(1)  # single row
            else:
                local_classes = np.arange(coef.shape[0])

        # 1D coef (e.g., Lasso/regression, or degenerate case)
        if coef.ndim == 1:
            m = min(self.n_features_, coef.shape[0])
            V[:, :m] = np.abs(coef[:m])[None, :]
            return V

        # 2D coef: (n_rows, n_features)
        n_rows, n_feat = coef.shape

        for c_local, c_val in enumerate(local_classes):
            # map local class to global index
            if hasattr(self, "classes_"):
                matches = np.where(self.classes_ == c_val)[0]
                if matches.size == 0:
                    continue
                g_idx = int(matches[0])
            else:
                if c_val < 0 or c_val >= K:
                    continue
                g_idx = int(c_val)

            if g_idx < 0 or g_idx >= K:
                continue

            # Robust to binary-logistic case: coef has only 1 row
            if c_local < n_rows:
                row = coef[c_local]
            else:
                # use the first row as a proxy for all remaining local classes
                row = coef[0]

            V[g_idx, :n_feat] = np.abs(row)

        return V

    def _compute_stfi_per_class(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-class STFI:
          For each class k: sum over leaves of w_leaf(ell) * (path_gains + |coef_k(ell)|)
        Returns:
          - stfi_class_raw:   array (K, p)
          - stfi_class_norm:  array (K, p) with row-wise normalization
        """
        if self.task_ not in ("binary", "multiclass"):
            # for safety; shouldn't be called in regression
            return (np.zeros((0, self.n_features_), dtype=float),
                    np.zeros((0, self.n_features_), dtype=float))

        K = self.n_classes_
        stfi_class = np.zeros((K, self.n_features_), dtype=float)

        w_leaf, _ = self._compute_leaf_support_weights()
        reachable = set(self._reachable_node_indices())

        for idx, n in enumerate(self.nodes_):
            if (idx not in reachable) or (not n.is_leaf) or (n.leaf_model is None):
                continue

            v_path = self._path_feature_vector_for_leaf(idx)    # (p,)
            V_coef = self._leaf_coef_magnitude_per_class(n.leaf_model)  # (K, p)
            w = w_leaf.get(idx, 0.0)

            # add contribution for each class
            # v_leaf_k = v_path + V_coef[k]
            stfi_class += w * (V_coef + v_path[None, :])

        # Normalize per class (row-wise)
        stfi_class_norm = stfi_class.copy()
        row_sums = stfi_class_norm.sum(axis=1, keepdims=True)
        nonzero = row_sums.squeeze() > 0
        stfi_class_norm[nonzero] /= row_sums[nonzero]

        return stfi_class, stfi_class_norm