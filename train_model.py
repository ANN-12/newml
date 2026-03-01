"""
train.py — 7 ML models | 33 features
Sentence: "the quick brown fox jumps over the lazy dog"

FIXES APPLIED:
  1. Outlier removal (total_duration, dwell_mean, flight_mean)
  2. Median instead of mean for zero value imputation
  3. Metadata.json saved after training
  4. Input validation on features

Run : python train.py
Deps: pip install scikit-learn imbalanced-learn pandas numpy
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import warnings
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.neural_network  import MLPClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.metrics         import classification_report, accuracy_score
from imblearn.over_sampling  import SMOTE

DATA_FILE = "keystrokes.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 33 features ──
ALL_FEATURES = [
    # Global timing
    "dwell_mean", "dwell_std", "dwell_cv",
    "flight_mean", "flight_std", "flight_cv",
    "timing_entropy", "total_duration", "wpm",
    # Digraphs
    "dg_th", "dg_he", "dg_qu", "dg_br", "dg_ow",
    "dg_fo", "dg_ju", "dg_ov", "dg_er", "dg_la", "dg_sp",
    # Trigraphs
    "tg_the", "tg_bro", "tg_own", "tg_ove", "tg_ver",
    # Key-specific dwell
    "kd_e", "kd_o", "kd_t", "kd_h", "kd_r", "kd_u", "kd_space",
    # Error
    "backspace_rate",
]

COL_MAP = {
    'userid':         'user_id',
    'dwellmean':      'dwell_mean',
    'dwellstd':       'dwell_std',
    'dwellcv':        'dwell_cv',
    'flightmean':     'flight_mean',
    'flightstd':      'flight_std',
    'flightcv':       'flight_cv',
    'timingentropy':  'timing_entropy',
    'totalduration':  'total_duration',
    'dgth':  'dg_th',  'dghe': 'dg_he', 'dgqu': 'dg_qu',
    'dgbr':  'dg_br',  'dgow': 'dg_ow', 'dgfo': 'dg_fo',
    'dgju':  'dg_ju',  'dgov': 'dg_ov', 'dger': 'dg_er',
    'dgla':  'dg_la',  'dgsp': 'dg_sp',
    'tgthe': 'tg_the', 'tgbro': 'tg_bro', 'tgown': 'tg_own',
    'tgove': 'tg_ove', 'tgver': 'tg_ver',
    'kde':   'kd_e',   'kdo':   'kd_o',   'kdt':   'kd_t',
    'kdh':   'kd_h',   'kdr':   'kd_r',   'kdu':   'kd_u',
    'kdspace': 'kd_space',
    'backspacerate': 'backspace_rate',
}

MODEL_CONFIGS = {
    "RandomForest": {
        "features": [
            "dwell_mean", "dwell_std", "flight_mean", "flight_std",
            "dg_th", "dg_he", "dg_er", "dg_sp",
            "tg_the", "kd_e", "kd_t", "timing_entropy"
        ],
        "boot_seed": 10,
        "model": RandomForestClassifier(
            n_estimators=300, max_depth=12,
            class_weight="balanced", random_state=42, n_jobs=-1)
    },
    "ExtraTrees": {
        "features": [
            "dwell_mean", "dwell_cv", "flight_mean", "flight_cv",
            "dg_th", "dg_qu", "dg_br", "dg_ow",
            "tg_bro", "kd_h", "kd_u", "wpm"
        ],
        "boot_seed": 20,
        "model": ExtraTreesClassifier(
            n_estimators=300, max_depth=12,
            class_weight="balanced", random_state=43, n_jobs=-1)
    },
    "GradientBoosting": {
        "features": [
            "dwell_mean", "flight_mean", "timing_entropy",
            "dg_he", "dg_ov", "dg_er", "dg_la",
            "tg_ove", "tg_ver", "kd_o", "kd_r", "wpm"
        ],
        "boot_seed": 30,
        "model": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=44)
    },
    "SVM_RBF": {
        "features": [
            "dwell_mean", "dwell_std", "flight_mean", "flight_std",
            "dg_th", "dg_he", "dg_sp", "dg_er",
            "tg_the", "kd_e", "kd_space", "timing_entropy"
        ],
        "boot_seed": 40,
        "model": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, class_weight="balanced", random_state=45)
    },
    "KNN": {
        "features": [
            "dwell_mean", "flight_mean",
            "dg_th", "dg_he", "dg_qu", "dg_sp",
            "tg_the", "timing_entropy"
        ],
        "boot_seed": 50,
        "model": None  # set dynamically after min_train known
    },
    "MLP": {
        "features": [
            "dwell_mean", "dwell_std", "dwell_cv",
            "flight_mean", "flight_std", "flight_cv",
            "dg_th", "dg_he", "dg_er", "dg_sp",
            "tg_the", "tg_own", "kd_e", "kd_t", "timing_entropy"
        ],
        "boot_seed": 60,
        "model": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            max_iter=2000, early_stopping=True,
            validation_fraction=0.15, random_state=46)
    },
    "GaussianNB": {
        "features": [
            "dwell_mean", "flight_mean", "timing_entropy",
            "dg_fo", "dg_ju", "dg_la", "dg_br",
            "tg_bro", "kd_o", "kd_r", "kd_space", "backspace_rate"
        ],
        "boot_seed": 70,
        "model": GaussianNB(var_smoothing=1e-8)
    },
}

# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD CSV
# ═══════════════════════════════════════════════════════════
print("Loading keystrokes.csv ...")
df = pd.read_csv(DATA_FILE)

# Drop timestamp if present
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

# Rename columns to standard names
df = df.rename(columns=COL_MAP)

# ═══════════════════════════════════════════════════════════
# FIX 1 — OUTLIER REMOVAL
# ═══════════════════════════════════════════════════════════
print("\n  Removing outliers ...")
before = len(df)

# Remove unrealistic session durations (> 60 seconds or < 3 seconds)
df = df[df["total_duration"] < 60000]
df = df[df["total_duration"] > 3000]

# Remove unrealistic dwell times
df = df[df["dwell_mean"] < 500]
df = df[df["dwell_mean"] > 10]

# Remove unrealistic flight times
df = df[df["flight_mean"] < 2000]
df = df[df["flight_mean"] > 0]

after = len(df)
print(f"  Removed {before - after} outlier rows  ({before} → {after} samples)")

# ═══════════════════════════════════════════════════════════
# STEP 2 — CLEAN NUMERIC FEATURES
# ═══════════════════════════════════════════════════════════
print("\n  Cleaning numeric features ...")
nan_before = df.isna().sum().sum()

for feat in ALL_FEATURES:
    if feat in df.columns:
        df[feat] = pd.to_numeric(df[feat], errors="coerce")
        if df[feat].isna().sum() > 0:
            df[feat] = df[feat].fillna(df[feat].median())

nan_after = df.isna().sum().sum()
print(f"  Fixed {nan_before} → {nan_after} NaN values")

# Drop rows still missing user_id or key features
df = df.dropna(subset=["user_id"] + [f for f in ALL_FEATURES if f in df.columns])

# ═══════════════════════════════════════════════════════════
# STEP 3 — PRINT SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  Loaded   : {len(df)} samples")
print(f"  Users    : {df['user_id'].nunique()}")

print(f"\n  Samples per user:")
for uid, cnt in df["user_id"].value_counts().items():
    bar    = "█" * min(cnt, 50)
    status = "✓" if cnt >= 50 else f"← need {50-cnt} more"
    print(f"    {uid:<12} {cnt:>3}  {bar}  {status}")

print(f"\n  Feature check (timing should be >10ms):")
for feat in ["dwell_mean", "flight_mean", "dg_th", "dg_he"]:
    if feat in df.columns:
        m    = df[feat].mean()
        flag = "✓" if m > 10 else "✗ TOO SMALL — check if seconds not ms!"
        print(f"    {feat:<18} mean={m:>8.2f} ms  {flag}")
print(f"{'='*60}\n")

# ═══════════════════════════════════════════════════════════
# FIX 2 — ZERO VALUE IMPUTATION USING MEDIAN (not mean)
# ═══════════════════════════════════════════════════════════
dg_tg_feats = [f for f in ALL_FEATURES if f.startswith(("dg_", "tg_", "kd_"))]

for feat in dg_tg_feats:
    if feat not in df.columns:
        continue
    for user in df["user_id"].unique():
        mask    = (df["user_id"] == user) & (df[feat] == 0)
        nonzero = df[(df["user_id"] == user) & (df[feat] > 0)][feat]
        if len(nonzero) > 0 and mask.sum() > 0:
            # FIX: use median instead of mean — robust to outliers
            df.loc[mask, feat] = nonzero.median()

# ═══════════════════════════════════════════════════════════
# STEP 4 — ENCODE LABELS & SPLIT
# ═══════════════════════════════════════════════════════════
le = LabelEncoder()
df["label"]  = le.fit_transform(df["user_id"])
available_feats = [f for f in ALL_FEATURES if f in df.columns]
X_all = df[available_feats].values
y_all = df["label"].values

print(f"  Using {len(available_feats)}/{len(ALL_FEATURES)} features: {available_feats[:5]}...")
print(f"{'='*60}\n")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=42
)
min_train = min(Counter(y_tr).values())

# Set KNN neighbors dynamically based on available data
MODEL_CONFIGS["KNN"]["model"] = KNeighborsClassifier(
    n_neighbors=max(3, min(7, min_train - 1)),
    weights="distance",
    metric="euclidean"
)

feat_idx = {f: i for i, f in enumerate(available_feats)}
def get_cols(names):
    return [feat_idx[f] for f in names if f in feat_idx]

# ═══════════════════════════════════════════════════════════
# STEP 5 — TRAIN 7 MODELS
# ═══════════════════════════════════════════════════════════
trained_models  = {}
model_scalers   = {}
model_feat_idxs = {}
accuracies      = {}   # FIX 3: track accuracy for metadata

print(f"{'='*60}")
print(f"  TRAINING — 7 independent models")
print(f"{'='*60}\n")

for name, cfg in MODEL_CONFIGS.items():
    col_idx   = get_cols(cfg["features"])
    boot_seed = cfg["boot_seed"]

    if len(col_idx) < 3:
        print(f"  [{name:<18}] SKIPPED — only {len(col_idx)} features available")
        continue

    # Bootstrap — each model sees a different random subset
    rng      = np.random.RandomState(boot_seed)
    boot_idx = rng.choice(len(X_tr), size=len(X_tr), replace=True)
    X_boot   = X_tr[boot_idx][:, col_idx]
    y_boot   = y_tr[boot_idx]

    # Each model has its own scaler — prevents data leakage
    sc        = StandardScaler()
    X_boot_sc = sc.fit_transform(X_boot)
    X_te_sc   = sc.transform(X_te[:, col_idx])

    # SMOTE — balance classes in training only
    min_boot = min(Counter(y_boot).values())
    k_smote  = max(1, min(5, min_boot - 1))
    try:
        sm = SMOTE(k_neighbors=k_smote, random_state=boot_seed)
        X_bal, y_bal = sm.fit_resample(X_boot_sc, y_boot)
    except Exception:
        X_bal, y_bal = X_boot_sc, y_boot

    # Calibrate probabilities for reliable confidence scores
    cal_cv     = min(3, min(Counter(y_bal).values()))
    calibrated = CalibratedClassifierCV(cfg["model"], method="isotonic", cv=cal_cv)
    calibrated.fit(X_bal, y_bal)

    preds = calibrated.predict(X_te_sc)
    acc   = accuracy_score(y_te, preds)
    accuracies[name] = round(float(acc), 4)

    print(f"  [{name:<18}] feats={len(col_idx):>2}  acc={acc:.3f}")
    print(classification_report(y_te, preds, target_names=le.classes_, zero_division=0))

    trained_models[name]  = calibrated
    model_scalers[name]   = sc
    model_feat_idxs[name] = col_idx

# ═══════════════════════════════════════════════════════════
# STEP 6 — ENSEMBLE AGREEMENT CHECK
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
if len(trained_models) > 1:
    agree = 0
    for i in range(len(X_te)):
        votes = set()
        for n in trained_models:
            p = trained_models[n].predict(
                model_scalers[n].transform(X_te[i:i+1, model_feat_idxs[n]])
            )[0]
            votes.add(int(p))
        if len(votes) == 1:
            agree += 1

    pct = agree / len(X_te) * 100
    print(f"  All {len(trained_models)} models agree: {agree}/{len(X_te)} ({pct:.1f}%)")
    print(f"  {'✓ GOOD — voting is meaningful' if pct < 70 else '⚠ High agreement — consider more data'}")
else:
    print("  ⚠️  Only 1 model trained — check your data")
print(f"{'='*60}\n")

# ═══════════════════════════════════════════════════════════
# STEP 7 — SAVE MODELS
# ═══════════════════════════════════════════════════════════
pickle.dump(trained_models,  open(f"{MODEL_DIR}/trained_models.pkl",  "wb"))
pickle.dump(model_scalers,   open(f"{MODEL_DIR}/model_scalers.pkl",   "wb"))
pickle.dump(le,              open(f"{MODEL_DIR}/label_encoder.pkl",   "wb"))
pickle.dump(model_feat_idxs, open(f"{MODEL_DIR}/model_feat_idxs.pkl","wb"))
pickle.dump(available_feats, open(f"{MODEL_DIR}/all_features.pkl",    "wb"))

# ═══════════════════════════════════════════════════════════
# FIX 3 — SAVE METADATA (who, when, how many, accuracy)
# ═══════════════════════════════════════════════════════════
metadata = {
    "trained_on":       datetime.now().isoformat(),
    "num_users":        int(len(le.classes_)),
    "users":            list(le.classes_),
    "total_samples":    int(len(df)),
    "features_used":    available_feats,
    "num_features":     len(available_feats),
    "test_size":        0.20,
    "model_accuracies": accuracies,
    "avg_accuracy":     round(float(np.mean(list(accuracies.values()))), 4),
    "best_model":       max(accuracies, key=accuracies.get),
    "best_accuracy":    max(accuracies.values()),
    "outlier_filters": {
        "total_duration_max_ms": 60000,
        "total_duration_min_ms": 3000,
        "dwell_mean_max_ms":     500,
        "flight_mean_max_ms":    2000,
    }
}

json.dump(metadata, open(f"{MODEL_DIR}/metadata.json", "w"), indent=2)

print(f"  ✓ Models saved to ./{MODEL_DIR}/")
print(f"  ✓ Metadata saved  → models/metadata.json")
print(f"\n  Users enrolled : {list(le.classes_)}")
print(f"  Best model     : {metadata['best_model']} ({metadata['best_accuracy']*100:.1f}%)")
print(f"  Avg accuracy   : {metadata['avg_accuracy']*100:.1f}%")
print(f"\n  → Run: python app.py\n")
