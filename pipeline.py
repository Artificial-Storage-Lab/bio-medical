"""
Shared pipeline components.

Everything reusable lives here: data loading, endpoint construction, feature
matrix assembly, the survival models, the per-fold selection routines, and the
nested cross-validation loop. The numbered scripts import from this module so
the same code path is used at every tier.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

import config as cfg

# Data loading and endpoint construction

def clean_placeholders(series):
    """Convert placeholder strings ('NA', 'NC', blanks) to real NaN.

    Several columns encode missingness as text. Left uncleaned, pandas treats
    these as valid categories, and .median() on such a column silently returns
    NaN, which makes imputation a no-op.
    """
    return series.apply(
        lambda x: np.nan if (isinstance(x, str) and x.strip() in cfg.PLACEHOLDER_VALUES) else x
    )


def load_clinical(path=None):
    """Load the patient table and clean placeholder strings."""
    path = path or cfg.PATIENT_TABLE_CSV
    df = pd.read_csv(path, low_memory=False)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = clean_placeholders(df[c])
    return df


def build_endpoints(df):
    """Construct overall survival and composite event-free survival endpoints.

    EFS event = death or recurrence, whichever came first. Overall survival
    alone yields too few events (62, 6.7%) to model reliably on this cohort.
    """
    death = [c for c in df.columns if "days_to_death" in c.lower()][0]
    last = [c for c in df.columns if "age_at_last_contact" in c.lower()][0]
    local = [c for c in df.columns if "days_to_local_recurrence" in c.lower()][0]
    distant = [c for c in df.columns if "days_to_distant_recurrence" in c.lower()][0]

    for c in (death, last, local, distant):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["os_event"] = df[death].notna().astype(int)
    df["os_time_days"] = df[death].fillna(df[last])

    composite = df[[death, local, distant]].min(axis=1)
    df["efs_event"] = composite.notna().astype(int)
    df["efs_time_days"] = composite.fillna(df["os_time_days"])

    return df

# Feature tier construction

CLINICAL_SPEC = {
    "age": ("date_of_birth_days", True),
    "stage_t": ("tumor_characteristics_stagingtumor_size_t", False),
    "stage_n": ("stagingnodesnx", False),
    "stage_m": ("stagingmetastasismx", False),
    "nottingham_grade": ("nottingham_grade", False),
    "histologic_type": ("histologic_type", False),
    "menopause": ("menopause_at_diagnosis", False),
    "race": ("race_and_ethnicity", False),
}

MOLECULAR_SPEC = {
    "er": ("tumor_characteristics_er", False),
    "pr": ("tumor_characteristics_pr", False),
    "her2": ("tumor_characteristics_her2", False),
    "mol_subtype": ("tumor_characteristics_mol_subtype", False),
    "oncotype_score": ("tumor_characteristics_oncotype_score", True),
}

TREATMENT_SPEC = {
    "chemo_neoadjuvant": ("chemotherapy_neoadjuvant_chemotherapy", False),
    "chemo_adjuvant": ("chemotherapy_adjuvant_chemotherapy", False),
    "endocrine_neoadjuvant": ("endocrine_therapy_neoadjuvant_endocrine_therapy_medications", False),
    "endocrine_adjuvant": ("endocrine_therapy_adjuvant_endocrine_therapy_medications", False),
    "anti_her2_neoadjuvant": ("anti_her2_neu_therapy_neoadjuvant_anti_her2_neu_therapy_", False),
    "anti_her2_adjuvant": ("anti_her2_neu_therapy_adjuvant_anti_her2_neu_therapy__", False),
    "radiation_neoadjuvant": ("radiation_therapy_neoadjuvant_radiation_therapy", False),
    "radiation_adjuvant": ("radiation_therapy_adjuvant_radiation_therapy", False),
    "surgery_type": ("surgery_definitive_surgery_type", False),
}


def _find_col(df, keyword):
    """Locate a column by keyword, excluding contralateral-breast duplicates."""
    matches = [c for c in df.columns
               if keyword in c.lower() and "other_side" not in c.lower()]
    return matches[0] if matches else None


def build_feature_block(clinical_df, spec, base=None):
    """Build an encoded feature block from a spec, optionally merged onto a base.

    Numeric fields are coerced; categorical fields are one-hot encoded. Columns
    with substantial missingness (grade, histologic type, Oncotype) get an
    explicit missing-indicator, since whether a test was ordered can itself be
    informative.
    """
    selected = {}
    for label, (keyword, _) in spec.items():
        col = _find_col(clinical_df, keyword)
        if col is not None:
            selected[label] = col

    out = clinical_df[[cfg.ID_COL] + list(selected.values())].copy()
    numeric_labels = {l for l, (_, is_num) in spec.items() if is_num}
    categorical = [l for l in selected if l not in numeric_labels]

    if "age" in selected:
        # stored as negative days from diagnosis
        out["age_at_diagnosis"] = pd.to_numeric(out[selected["age"]], errors="coerce").abs() / 365.25
        out = out.drop(columns=[selected["age"]])
        categorical = [c for c in categorical if c != "age"]
        del selected["age"]

    if "oncotype_score" in selected:
        col = selected["oncotype_score"]
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_missing"] = out[col].isna().astype(int)
        categorical = [c for c in categorical if c != "oncotype_score"]

    for label in ("nottingham_grade", "histologic_type"):
        if label in selected:
            out[f"{selected[label]}_missing"] = out[selected[label]].isna().astype(int)

    cat_cols = [selected[l] for l in categorical if l in selected]
    out = pd.get_dummies(out, columns=cat_cols, dummy_na=False)

    return out if base is None else base.merge(out, on=cfg.ID_COL, how="inner")


def feature_columns(df):
    """Feature columns, excluding the ID and outcome columns."""
    return [c for c in df.columns if c not in cfg.OUTCOME_COLS and c != cfg.ID_COL]


def coerce_numeric(df, cols):
    """Force numeric dtype and median-impute. Applied before any model fitting."""
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med if not pd.isna(med) else 0.0)
    return df

# Per-fold feature selection (training data only)

def remove_near_zero_variance(df, cols):
    return [c for c in cols if df[c].nunique(dropna=True) > 1]


def univariate_cox_pvalues(df, cols, time_col, event_col):
    """Univariate Cox p-value for each candidate feature."""
    from lifelines import CoxPHFitter

    pvalues = {}
    for col in cols:
        sub = df[[time_col, event_col, col]].dropna()
        sub = sub.rename(columns={time_col: "time", event_col: "event", col: "feat"})
        if len(sub) < 10 or sub["feat"].std() == 0:
            pvalues[col] = 1.0
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col="time", event_col="event")
            pvalues[col] = cph.summary.loc["feat", "p"]
        except Exception:
            pvalues[col] = 1.0
    return pvalues


def remove_correlated(df, ranked_cols, threshold=None):
    """Greedy decorrelation over a significance-ranked list.

    Candidates are visited most-significant-first, so when two features are
    highly correlated the one with the stronger univariate association is kept.
    """
    threshold = threshold or cfg.CORRELATION_THRESHOLD
    corr = df[ranked_cols].corr().abs()
    kept = []
    for col in ranked_cols:
        if all(corr.loc[col, k] < threshold for k in kept):
            kept.append(col)
    return kept


def lasso_cox_select(df, cols, time_col, event_col, target_range=None):
    """LASSO-Cox selection, returning features with non-zero coefficients."""
    target_range = target_range or cfg.LASSO_TARGET_RANGE
    sub = df[[time_col, event_col] + cols].dropna()
    if len(sub) < 20:
        return cols[:target_range[1]]

    X = StandardScaler().fit_transform(sub[cols].values)
    y = Surv.from_arrays(event=sub[event_col].values.astype(bool),
                         time=sub[time_col].values)
    try:
        model = CoxnetSurvivalAnalysis(l1_ratio=0.99, alpha_min_ratio=0.01,
                                       n_alphas=50, max_iter=10000)
        model.fit(X, y)
        coefs = model.coef_
        best_idx, best_diff = 0, np.inf
        for i in range(coefs.shape[1]):
            n_nonzero = int(np.sum(coefs[:, i] != 0))
            if target_range[0] <= n_nonzero <= target_range[1]:
                best_idx = i
                break
            diff = min(abs(n_nonzero - target_range[0]), abs(n_nonzero - target_range[1]))
            if diff < best_diff:
                best_diff, best_idx = diff, i
        selected = [cols[j] for j in range(len(cols)) if coefs[j, best_idx] != 0]
        return selected if selected else cols[:target_range[0]]
    except Exception:
        variances = sub[cols].var(axis=0, skipna=True)
        return variances.sort_values(ascending=False).head(target_range[1]).index.tolist()


def select_radiomics_per_fold(df, train_pos, candidate_cols,
                              time_col="efs_time_days", event_col="efs_event"):
    """Full radiomics selection pipeline, fit on the training fold only.

    near-zero variance removal -> univariate Cox ranking -> correlation removal
    -> top-K truncation -> LASSO-Cox
    """
    train = df.iloc[train_pos]
    step1 = remove_near_zero_variance(train, candidate_cols)
    pvalues = univariate_cox_pvalues(train, step1, time_col, event_col)
    ranked = sorted(pvalues, key=lambda c: pvalues[c])
    step2 = remove_correlated(train, ranked)
    step3 = step2[:cfg.UNIVARIATE_TOP_K]
    return lasso_cox_select(train, step3, time_col, event_col)


def select_top_n_by_significance(df, train_pos, candidate_cols, n_final,
                                 time_col="efs_time_days", event_col="efs_event"):
    """Deterministic top-N selection, for experiments that need an exact count.

    LASSO chooses its own sparsity, so it cannot be forced to return exactly N
    features. This variant stops after the correlation-removal step instead.
    """
    train = df.iloc[train_pos]
    step1 = remove_near_zero_variance(train, candidate_cols)
    pvalues = univariate_cox_pvalues(train, step1, time_col, event_col)
    ranked = sorted(pvalues, key=lambda c: pvalues[c])
    return remove_correlated(train, ranked)[:n_final]

# DeepSurv

class DeepSurvNet(nn.Module):
    """Two hidden layers, SELU activation, AlphaDropout.

    AlphaDropout rather than standard dropout: SELU is self-normalising, and
    ordinary dropout destroys that property.
    """

    def __init__(self, in_features, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.SELU(), nn.AlphaDropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SELU(), nn.AlphaDropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ModifiedDeepSurvNet(nn.Module):
    """Reference configuration from Lei et al. (2023).

    One hidden layer of 450 units, SELU, single tanh output. This is DeepSurv
    with fixed published hyperparameters, not a distinct architecture.
    """

    def __init__(self, in_features, hidden=450):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.SELU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cox_ph_loss(risk, time, event):
    """Negative partial log-likelihood, Breslow approximation."""
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    return -torch.sum((risk - log_cumsum) * event) / (event.sum() + 1e-8)


def train_deepsurv(X_train, time_train, event_train, X_val, time_val, event_val,
                   hidden=64, lr=1e-3, dropout=0.3, epochs=150, patience=20):
    """Train with early stopping.

    The validation split must come from training data. Passing the test fold
    here would select the best epoch using held-out data, which inflates the
    reported score.
    """
    torch.manual_seed(cfg.RANDOM_STATE)
    model = DeepSurvNet(X_train.shape[1], hidden=hidden, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    tt = torch.tensor(time_train, dtype=torch.float32)
    et = torch.tensor(event_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)

    best_c, best_state, stale = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = cox_ph_loss(model(Xt), tt, et)
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_risk = model(Xv).numpy()
        if np.isnan(val_risk).any():
            break
        try:
            c = concordance_index_censored(event_val.astype(bool), time_val, val_risk)[0]
        except Exception:
            c = 0.5
        if c > best_c:
            best_c, best_state, stale = c, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_modified_deepsurv(X_train, time_train, event_train,
                            hidden=450, lr=0.07, decay=0.003, epochs=200):
    """Train the reference configuration exactly as published.

    No early stopping and no hyperparameter search -- that is the point of this
    baseline. Returns (model, diverged); lr=0.07 is aggressive for Adam and can
    produce NaN loss on some folds.
    """
    torch.manual_seed(cfg.RANDOM_STATE)
    model = ModifiedDeepSurvNet(X_train.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - decay)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    tt = torch.tensor(time_train, dtype=torch.float32)
    et = torch.tensor(event_train, dtype=torch.float32)

    diverged = False
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = cox_ph_loss(model(Xt), tt, et)
        if torch.isnan(loss):
            diverged = True
            break
        loss.backward()
        optimizer.step()
        scheduler.step()

    return model, diverged


def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()

# Fold assembly

def impute_from_train(X_train, X_test):
    """Median-impute both splits using training-fold medians only."""
    X_train, X_test = X_train.copy(), X_test.copy()
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        med = np.nanmedian(X_train, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        i = np.where(np.isnan(X_train))
        X_train[i] = np.take(med, i[1])
        j = np.where(np.isnan(X_test))
        X_test[j] = np.take(med, j[1])
    return X_train, X_test


def build_fold_matrices(df, train_pos, test_pos, fixed_cols,
                        radiomics_cols=None, emb_cols=None,
                        pca_n=None, radiomics_selector=None):
    """Assemble train/test matrices for one fold.

    Radiomics selection and embedding PCA are both fit on training patients
    only, then applied to the test fold.
    """
    pca_n = pca_n or cfg.PCA_N_COMPONENTS
    blocks_train = [df.iloc[train_pos][fixed_cols].values.astype(float)]
    blocks_test = [df.iloc[test_pos][fixed_cols].values.astype(float)]
    selected_radiomics = None

    if radiomics_cols is not None:
        selector = radiomics_selector or select_radiomics_per_fold
        selected_radiomics = selector(df, train_pos, radiomics_cols)
        blocks_train.append(df.iloc[train_pos][selected_radiomics].values.astype(float))
        blocks_test.append(df.iloc[test_pos][selected_radiomics].values.astype(float))

    if emb_cols is not None:
        emb = df[emb_cols].values.astype(float)
        scaler = StandardScaler()
        e_train = scaler.fit_transform(np.nan_to_num(emb[train_pos]))
        e_test = scaler.transform(np.nan_to_num(emb[test_pos]))
        pca = PCA(n_components=min(pca_n, len(train_pos) - 1),
                  random_state=cfg.RANDOM_STATE)
        blocks_train.append(pca.fit_transform(e_train))
        blocks_test.append(pca.transform(e_test))

    X_train, X_test = impute_from_train(np.hstack(blocks_train), np.hstack(blocks_test))
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), selected_radiomics

# Nested cross-validation

def _tune_coxph(X, time, event, inner_cv):
    best_alpha, best_score = 1.0, -np.inf
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        scores = []
        for tr, va in inner_cv.split(X, event):
            try:
                m = CoxPHSurvivalAnalysis(alpha=alpha)
                m.fit(X[tr], Surv.from_arrays(event=event[tr], time=time[tr]))
                scores.append(concordance_index_censored(event[va], time[va], m.predict(X[va]))[0])
            except Exception:
                scores.append(0.5)
        if np.mean(scores) > best_score:
            best_score, best_alpha = np.mean(scores), alpha
    return best_alpha


def _tune_rsf(X, time, event, inner_cv):
    best_params, best_score = {"n_estimators": 200, "max_depth": 5}, -np.inf
    for n_est in [100, 200]:
        for depth in [3, 5, None]:
            scores = []
            for tr, va in inner_cv.split(X, event):
                try:
                    m = RandomSurvivalForest(n_estimators=n_est, max_depth=depth,
                                             random_state=cfg.RANDOM_STATE, n_jobs=-1)
                    m.fit(X[tr], Surv.from_arrays(event=event[tr], time=time[tr]))
                    scores.append(concordance_index_censored(event[va], time[va], m.predict(X[va]))[0])
                except Exception:
                    scores.append(0.5)
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_params = {"n_estimators": n_est, "max_depth": depth}
    return best_params


def _tune_gb(X, time, event, inner_cv):
    best_params, best_score = {"n_estimators": 100, "learning_rate": 0.1}, -np.inf
    for n_est in [100, 200]:
        for lr in [0.05, 0.1]:
            scores = []
            for tr, va in inner_cv.split(X, event):
                try:
                    m = GradientBoostingSurvivalAnalysis(
                        n_estimators=n_est, learning_rate=lr, max_depth=3,
                        random_state=cfg.RANDOM_STATE)
                    m.fit(X[tr], Surv.from_arrays(event=event[tr], time=time[tr]))
                    scores.append(concordance_index_censored(event[va], time[va], m.predict(X[va]))[0])
                except Exception:
                    scores.append(0.5)
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_params = {"n_estimators": n_est, "learning_rate": lr}
    return best_params


def _tune_deepsurv(X, time, event, inner_cv, hidden_sizes=(16, 32, 64)):
    best_hidden, best_score = 32, -np.inf
    for hidden in hidden_sizes:
        scores = []
        for tr, va in inner_cv.split(X, event):
            m = train_deepsurv(X[tr], time[tr], event[tr], X[va], time[va], event[va],
                               hidden=hidden, epochs=100, patience=15)
            pred = predict_torch(m, X[va])
            if np.isnan(pred).any():
                scores.append(0.5)
                continue
            try:
                scores.append(concordance_index_censored(event[va], time[va], pred)[0])
            except Exception:
                scores.append(0.5)
        if np.mean(scores) > best_score:
            best_score, best_hidden = np.mean(scores), hidden
    return best_hidden


def run_nested_cv(df, fixed_cols, tier_name, radiomics_cols=None, emb_cols=None,
                  return_predictions=False, deepsurv_hidden_sizes=(16, 32, 64)):
    """Nested cross-validation over four model families.

    Outer folds estimate performance; inner folds select hyperparameters. All
    preprocessing, feature selection and tuning happen inside the training
    portion of each outer fold.
    """
    time_all = df["efs_time_days"].values
    event_all = df["efs_event"].values.astype(bool)
    positions = np.arange(len(df))

    print(f"\n{'=' * 70}\n{tier_name}: N={len(df)}, fixed features={len(fixed_cols)}"
          + (f", radiomics candidates={len(radiomics_cols)}" if radiomics_cols else "")
          + (f", embedding dims={len(emb_cols)}" if emb_cols else "")
          + f"\n{'=' * 70}")

    outer_cv = StratifiedKFold(n_splits=cfg.OUTER_FOLDS, shuffle=True,
                               random_state=cfg.RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=cfg.INNER_FOLDS, shuffle=True,
                               random_state=cfg.RANDOM_STATE)

    results = {"CoxPH": [], "RSF": [], "GradientBoosting": [], "DeepSurv": [], "Ensemble": []}
    selection_history = []
    oof_risk = np.full(len(df), np.nan)

    for fold, (train_pos, test_pos) in enumerate(outer_cv.split(positions, event_all)):
        X_train, X_test, selected = build_fold_matrices(
            df, train_pos, test_pos, fixed_cols, radiomics_cols, emb_cols)
        if selected is not None:
            selection_history.append(selected)

        time_train, time_test = time_all[train_pos], time_all[test_pos]
        event_train, event_test = event_all[train_pos], event_all[test_pos]
        y_train = Surv.from_arrays(event=event_train, time=time_train)
        fold_preds = {}

        alpha = _tune_coxph(X_train, time_train, event_train, inner_cv)
        cox = CoxPHSurvivalAnalysis(alpha=alpha)
        cox.fit(X_train, y_train)
        fold_preds["CoxPH"] = cox.predict(X_test)

        rsf_params = _tune_rsf(X_train, time_train, event_train, inner_cv)
        rsf = RandomSurvivalForest(random_state=cfg.RANDOM_STATE, n_jobs=-1, **rsf_params)
        rsf.fit(X_train, y_train)
        fold_preds["RSF"] = rsf.predict(X_test)

        gb_params = _tune_gb(X_train, time_train, event_train, inner_cv)
        gb = GradientBoostingSurvivalAnalysis(max_depth=3, random_state=cfg.RANDOM_STATE, **gb_params)
        gb.fit(X_train, y_train)
        fold_preds["GradientBoosting"] = gb.predict(X_test)

        hidden = _tune_deepsurv(X_train, time_train, event_train, inner_cv, deepsurv_hidden_sizes)
        fit_idx, es_idx = train_test_split(
            np.arange(len(X_train)), test_size=0.2,
            stratify=event_train, random_state=cfg.RANDOM_STATE)
        ds = train_deepsurv(X_train[fit_idx], time_train[fit_idx], event_train[fit_idx],
                            X_train[es_idx], time_train[es_idx], event_train[es_idx],
                            hidden=hidden)
        ds_pred = predict_torch(ds, X_test)
        fold_preds["DeepSurv"] = ds_pred if not np.isnan(ds_pred).any() else np.zeros(len(X_test))

        for name, pred in fold_preds.items():
            try:
                results[name].append(concordance_index_censored(event_test, time_test, pred)[0])
            except Exception:
                results[name].append(0.5)

        # Rank-average ensemble: the four models output risk on different
        # scales, so raw averaging would let one dominate.
        from scipy.stats import rankdata
        ranks = [rankdata(p) for p in fold_preds.values() if not np.isnan(p).any()]
        ensemble_pred = np.mean(ranks, axis=0)
        results["Ensemble"].append(
            concordance_index_censored(event_test, time_test, ensemble_pred)[0])

        oof_risk[test_pos] = fold_preds["CoxPH"]

        print(f"  Fold {fold + 1}: " + " | ".join(
            f"{k}={results[k][-1]:.3f}" for k in ["CoxPH", "RSF", "GradientBoosting", "DeepSurv", "Ensemble"]))

    print(f"\n{tier_name} summary:")
    for name, scores in results.items():
        mean, lo, hi = bootstrap_ci(scores)
        print(f"  {name:<18} {mean:.3f} +/- {np.std(scores):.3f}   [95% CI {lo:.3f}, {hi:.3f}]")

    if return_predictions:
        return results, selection_history, oof_risk
    return results, selection_history

# Statistics and persistence

def bootstrap_ci(fold_scores, n_boot=2000, alpha=0.05):
    """Bootstrap confidence interval over per-fold scores."""
    scores = np.array(fold_scores)
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    boot = [np.mean(rng.choice(scores, size=len(scores), replace=True)) for _ in range(n_boot)]
    return float(np.mean(scores)), float(np.percentile(boot, 100 * alpha / 2)), \
        float(np.percentile(boot, 100 * (1 - alpha / 2)))


def save_results(name, results, extra=None):
    """Write per-fold scores to JSON so a lost session never costs a re-run."""
    cfg.ensure_dirs()
    payload = {"fold_scores": {k: list(map(float, v)) for k, v in results.items()},
               "summary": {}}
    for model, scores in results.items():
        mean, lo, hi = bootstrap_ci(scores)
        payload["summary"][model] = {"mean": mean, "std": float(np.std(scores)),
                                     "ci_low": lo, "ci_high": hi}
    if extra:
        payload.update(extra)

    path = os.path.join(cfg.RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {path}")
    return path


def load_results(name):
    path = os.path.join(cfg.RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
