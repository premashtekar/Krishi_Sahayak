#!/usr/bin/env python3
"""
Retrain crop model with automatic oversampling to fix dominant-class predictions.
Saves artifacts to ./models/ (same as Streamlit app expects).

Usage:
    python retrain_balance.py
"""
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

# --- CONFIG (match app paths) ---
DATA_DIR = "data"
CROP_FILE_NAME = "Crop_recommendation.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CROP_FILE = os.path.join(DATA_DIR, CROP_FILE_NAME)
CROP_MODEL_FILE = os.path.join(MODEL_DIR, "rf_crop_model.joblib")
CROP_LBL_FILE = os.path.join(MODEL_DIR, "crop_label_encoder.joblib")
CROP_FEATS_FILE = os.path.join(MODEL_DIR, "crop_features.joblib")
CROP_ENC_MAP_FILE = os.path.join(MODEL_DIR, "crop_enc_map.joblib")

# --- Helpers (same heuristics as app) ---
def detect_target_column(df):
    for cand in ['crop','Crop','CROP','recommended_crop','label','target']:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if df[c].dtype == object and 2 <= df[c].nunique() <= 200:
            return c
    return df.columns[-1]

def safe_save(obj, path):
    joblib.dump(obj, path)
    print("Saved:", path)

# --- Main retrain pipeline ---
def main():
    if not os.path.exists(CROP_FILE):
        raise SystemExit(f"Crop file not found at {CROP_FILE}. Place your CSV at that path.")

    df = pd.read_csv(CROP_FILE)
    print("Loaded CSV shape:", df.shape)

    target = detect_target_column(df)
    print("Detected target column:", target)

    # drop rows with no target
    df = df.dropna(subset=[target]).copy()

    # exclude obvious IDs/coords
    exclude = {target, 'id','Id','ID','latitude','longitude','lat','lon'}
    feats = [c for c in df.columns if c not in exclude and df[c].nunique() > 1]
    feats = [c for c in feats if not (df[c].dtype == object and df[c].nunique() > 300)]

    if not feats:
        raise SystemExit("No usable features found. Check your CSV columns.")

    print("Using features (count):", len(feats))
    print("Features sample:", feats[:30])

    # Prepare X, y
    y_raw = df[target].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X = df[feats].copy()

    # Build encoders for categorical features and transform X
    enc_map = {}
    for c in X.columns:
        if X[c].dtype == object or X[c].dtype == 'category':
            X[c] = X[c].fillna("NA").astype(str)
            enc = LabelEncoder()
            try:
                X[c] = enc.fit_transform(X[c])
                enc_map[c] = enc
            except Exception:
                X[c] = X[c].apply(lambda v: abs(hash(str(v))) % 1000)
        else:
            X[c] = X[c].fillna(X[c].median())

    # Class distribution
    vc = pd.Series(y).value_counts()
    class_names = le.inverse_transform(vc.index.tolist())
    print("Class distribution (top 10):")
    for name, cnt in zip(class_names[:10], vc.iloc[:10].tolist()):
        print(f"  {name}: {cnt}")
    print("Total unique classes:", len(vc))

    if len(np.unique(y)) < 2:
        raise SystemExit("Only one class present in dataset — cannot train a classifier.")

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Oversampling minority classes in the training set to match majority
    train_df = X_train.copy()
    train_df['_y'] = y_train
    majority_count = train_df['_y'].value_counts().max()
    frames = []
    for cls, grp in train_df.groupby('_y'):
        if len(grp) < majority_count:
            upsampled = resample(grp, replace=True, n_samples=majority_count, random_state=42)
            frames.append(upsampled)
            print(f"  Upsampled class {cls} from {len(grp)} to {len(upsampled)}")
        else:
            frames.append(grp)
            print(f"  Kept class {cls} at {len(grp)}")
    train_bal = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
    y_train_bal = train_bal['_y'].values
    X_train_bal = train_bal.drop(columns=['_y']).values

    print("Training samples after balancing:", X_train_bal.shape[0])

    # Train RandomForest with balanced class weight
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train_bal, y_train_bal)

    # Evaluate on held-out test set
    preds = model.predict(X_test)
    rep = classification_report(y_test, preds, zero_division=0)
    print("Classification report on test split:\n")
    print(rep)

    # Save artifacts (overwrite)
    safe_save(model, CROP_MODEL_FILE)
    safe_save(le, CROP_LBL_FILE)
    safe_save(feats, CROP_FEATS_FILE)
    safe_save(enc_map, CROP_ENC_MAP_FILE)

    print("Retraining complete. Models saved to ./models/")

if __name__ == "__main__":
    main()
