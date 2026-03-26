

# Input:
#   X          : (n_bouts, 12) array of bout-averaged postural features.
#                In torpor vs sleep classification, only bouts assigned to the curled-up cluster are included.
#                In torpor vs cold exposure classification, all immobility-defined bouts are included.
#   y          : binary labels (1 = torpor, 0 = comparison state)
#   animal_ids : animal identifier for leave-one-animal-out cross-validation

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut


THRESHOLD_PCT = 90.0
THR_PRED = 0.5
N_SHUFFLES = 100
SEED_UNDER = 0
SEED_SHUF = 123


def mean_sem(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return arr.mean(), np.nan
    return arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr))


def undersample_binary(X, y, rng):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y

    n_min = counts.min()
    keep_idx = []

    for c in classes:
        idx_c = np.where(y == c)[0]
        chosen = rng.choice(idx_c, size=n_min, replace=False)
        keep_idx.append(chosen)

    keep_idx = np.concatenate(keep_idx)
    return X[keep_idx], y[keep_idx]


def select_n_pcs_by_cumulative_variance(X_train_scaled, threshold_pct=THRESHOLD_PCT):
    n_feat = X_train_scaled.shape[1]
    n_comp_full = min(n_feat, X_train_scaled.shape[0])

    pca_full = PCA(n_components=n_comp_full)
    X_train_pca_full = pca_full.fit_transform(X_train_scaled)

    evr = pca_full.explained_variance_ratio_
    cum_evr_pct = np.cumsum(evr) * 100.0

    k_keep = int(np.searchsorted(cum_evr_pct, threshold_pct) + 1)
    k_keep = max(1, min(k_keep, n_comp_full))

    return pca_full, X_train_pca_full, cum_evr_pct, k_keep


def logistic_regression_loo_cv(
    X,
    y,
    animal_ids,
    threshold_pct=THRESHOLD_PCT,
    thr_pred=THR_PRED,
    n_shuffles=N_SHUFFLES,
    seed_under=SEED_UNDER,
    seed_shuf=SEED_SHUF,
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    animal_ids = np.asarray(animal_ids)

    rng_under = np.random.default_rng(seed_under)
    rng_shuf = np.random.default_rng(seed_shuf)

    logo = LeaveOneGroupOut()

    p_torpor = np.zeros(len(X), dtype=float)
    fold_rows = []
    abs_feature_weights = []

    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=animal_ids), start=1):
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train_raw = y[train_idx]
        y_test = y[test_idx]

        test_animal = animal_ids[test_idx][0]

        X_train_use, y_train_use = undersample_binary(X_train_raw, y_train_raw, rng_under)

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_use)
        X_test_sc = scaler.transform(X_test_raw)

        pca_full, X_train_pca_full, cum_evr_pct, k_keep = select_n_pcs_by_cumulative_variance(
            X_train_sc,
            threshold_pct=threshold_pct,
        )
        X_test_pca_full = pca_full.transform(X_test_sc)

        X_train_pca = X_train_pca_full[:, :k_keep]
        X_test_pca = X_test_pca_full[:, :k_keep]

        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
        )
        clf.fit(X_train_pca, y_train_use)

        proba_test = clf.predict_proba(X_test_pca)[:, 1]
        p_torpor[test_idx] = proba_test

        y_pred_test = (proba_test >= thr_pred).astype(int)
        acc_real = (y_pred_test == y_test).mean()

        if len(np.unique(y_train_use)) < 2:
            acc_shuf = np.nan
        else:
            shuf_accs = []
            for _ in range(n_shuffles):
                y_shuf = y_train_use.copy()
                rng_shuf.shuffle(y_shuf)

                clf_shuf = LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                )
                clf_shuf.fit(X_train_pca, y_shuf)

                proba_shuf = clf_shuf.predict_proba(X_test_pca)[:, 1]
                y_pred_shuf = (proba_shuf >= thr_pred).astype(int)
                shuf_accs.append((y_pred_shuf == y_test).mean())

            acc_shuf = float(np.mean(shuf_accs))

        coef_pc = clf.coef_.ravel()
        loadings = pca_full.components_[:k_keep, :]
        coef_feature = loadings.T @ coef_pc
        abs_feature_weights.append(np.abs(coef_feature))

        fold_rows.append(
            {
                "fold": fold_i,
                "animal_test": test_animal,
                "k_keep": k_keep,
                "cumEV_at_k_keep_pct": float(cum_evr_pct[k_keep - 1]),
                "acc_real": float(acc_real),
                "acc_shuf": acc_shuf,
            }
        )

    df_fold = pd.DataFrame(fold_rows)
    mean_abs_standardized_weights = np.mean(np.vstack(abs_feature_weights), axis=0)

    return {
        "p_torpor": p_torpor,
        "df_fold": df_fold,
        "mean_abs_standardized_weights": mean_abs_standardized_weights,
    }
