# app.py

import os, json, datetime, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, mean_absolute_error
import plotly.graph_objects as go

# ---------------- Paths & config ----------------
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
PRICE_MODEL_DIR = MODELS_DIR / "price_models"
for d in (DATA_DIR, MODELS_DIR, PRICE_MODEL_DIR):
    os.makedirs(d, exist_ok=True)

FERT_DATA_FILE = DATA_DIR / "data_core.csv"    # expected fertilizer dataset
CROP_REC_FILE = DATA_DIR / "Crop_recommendation.csv"  # optional crop dataset
PRICE_FILE = DATA_DIR / "Crops_data.csv"

FERT_MODEL = MODELS_DIR / "fert_model.pkl"
FERT_LE = MODELS_DIR / "fert_label.pkl"
FERT_FEATS = MODELS_DIR / "fert_features.pkl"
FERT_META = MODELS_DIR / "fert_meta.json"

PRED_LOG = MODELS_DIR / "prediction_logs.csv"

SEED = 42
np.random.seed(SEED)

# ---------------- Domain mappings ----------------
FERTILIZER_TO_CROPS = {
    "Urea": ["Rice", "Wheat"],
    "DAP": ["Chickpea", "Green Gram"],
    "20-20-0": ["Maize"],
    "17-17-17": ["Tomato"],
    "28-28-0": ["Sugarcane"],
    "14-35-14": ["Groundnut"],
    "10-26-26": ["Potato"]
}

CROPS_DB = {
    "Rice":      {"temp":(20,32),"humidity":(70,100),"moisture":(60,100),"N":(30,60),"P":(20,40),"K":(25,50)},
    "Wheat":     {"temp":(10,25),"humidity":(40,70),"moisture":(40,70),"N":(20,50),"P":(15,30),"K":(20,40)},
    "Maize":     {"temp":(18,32),"humidity":(50,80),"moisture":(40,70),"N":(25,60),"P":(18,35),"K":(20,45)},
    "Tomato":    {"temp":(18,30),"humidity":(50,90),"moisture":(40,70),"N":(20,40),"P":(15,30),"K":(20,40)},
    "Sugarcane": {"temp":(20,35),"humidity":(60,90),"moisture":(60,90),"N":(30,60),"P":(20,40),"K":(30,60)},
    "Groundnut": {"temp":(20,32),"humidity":(50,80),"moisture":(30,60),"N":(15,35),"P":(10,25),"K":(15,30)},
    "Potato":    {"temp":(10,25),"humidity":(60,90),"moisture":(50,80),"N":(40,80),"P":(30,60),"K":(40,80)},
    "Chickpea":  {"temp":(15,30),"humidity":(40,70),"moisture":(20,50),"N":(10,30),"P":(10,25),"K":(10,25)}
}

SOIL_TYPES = ["Sandy","Loamy","Black","Red","Clayey"]

# ---------------- Helpers ----------------
def now_str(): return datetime.datetime.now().isoformat()
def save_json(o,p): json.dump(o, open(p,"w"), indent=2)
def load_json(p): return json.load(open(p,"r"))

def append_log(row:dict):
    df = pd.DataFrame([row])
    if not PRED_LOG.exists():
        df.to_csv(PRED_LOG, index=False)
    else:
        df.to_csv(PRED_LOG, mode='a', header=False, index=False)

# robust mapping input -> feature order
def map_features(features, user_inputs, defaults=None):
    mapped = {}
    missing = []
    ui_lower = {k.lower(): k for k in user_inputs.keys()}
    for f in features:
        if f in user_inputs:
            mapped[f]=user_inputs[f]; continue
        if f.lower() in ui_lower:
            mapped[f]=user_inputs[ui_lower[f.lower()]]; continue
        # variants
        for v in (f.replace(" ","").lower(), f.replace("_","").lower()):
            if v in ui_lower:
                mapped[f]=user_inputs[ui_lower[v]]; break
        if f in mapped: continue
        # default
        if defaults and f in defaults:
            mapped[f]=defaults[f]
        else:
            mapped[f]=0.0; missing.append(f)
    return mapped, missing

# suitability scoring helpers
def range_distance_score(val, rng):
    if val is None: return 0.0
    low,high = rng
    if low <= val <= high: return 1.0
    width = max(1.0, high-low)
    d = low-val if val<low else val-high
    rel = d/width
    if rel<=0.10: return 0.7
    if rel<=0.30: return 0.4
    return 0.0

def crop_suitability_score(crop, user_inputs):
    info = CROPS_DB.get(crop)
    if not info: return 0.0
    t = float(user_inputs.get("Temperature",0))
    h = float(user_inputs.get("Humidity",0))
    m = float(user_inputs.get("Moisture",0))
    n = float(user_inputs.get("Nitrogen",0))
    p = float(user_inputs.get("Phosphorous",0))
    k = float(user_inputs.get("Potassium",0))
    sc = 0.0
    sc += 0.22 * range_distance_score(t, info["temp"])
    sc += 0.18 * range_distance_score(h, info["humidity"])
    sc += 0.20 * range_distance_score(m, info["moisture"])
    sc += 0.13 * range_distance_score(n, info["N"])
    sc += 0.13 * range_distance_score(p, info["P"])
    sc += 0.14 * range_distance_score(k, info["K"])
    return sc

def recommend_crops_from_fert_and_env(fert_label, user_inputs, top_k=5):
    direct = []
    for k in FERTILIZER_TO_CROPS:
        if k.lower() == fert_label.lower() or k.lower() in fert_label.lower() or fert_label.lower() in k.lower():
            direct = FERTILIZER_TO_CROPS.get(k, [])
            break
    candidates = list(dict.fromkeys(direct + list(CROPS_DB.keys())))
    scored=[]
    for c in candidates:
        s = crop_suitability_score(c, user_inputs)
        if c in direct:
            s = min(1.0, s+0.08)
        scored.append((c,s))
    scored = sorted(scored, key=lambda x:x[1], reverse=True)
    filtered = [c for c,s in scored if s>=0.12]
    if not filtered:
        return [c for c,_ in scored][:top_k]
    final=[]
    for c,s in scored:
        if c=="Groundnut" and s<0.65:
            s = s*0.6
        final.append((c,s))
    final = sorted(final, key=lambda x:x[1], reverse=True)
    return [c for c,_ in final][:top_k]

# ---------------- Data & Training (fertilizer model) ----------------
def load_fert_dataset():
    if FERT_DATA_FILE.exists():
        try:
            df=pd.read_csv(FERT_DATA_FILE)
            return df
        except Exception:
            pass
    # synth fallback small
    st.warning("Fertilizer dataset not found; creating small synthetic dataset for demo.")
    rows=[]
    for fert,crops in FERTILIZER_TO_CROPS.items():
        for i in range(60):
            crop = crops[i % len(crops)]
            info = CROPS_DB.get(crop, {"temp":(20,30),"humidity":(40,70),"moisture":(30,60),"N":(20,40),"P":(10,30),"K":(15,35)})
            temp = np.random.uniform(*info["temp"]) + np.random.normal(0,1)
            hum = np.random.uniform(*info["humidity"]) + np.random.normal(0,2)
            moist = np.random.uniform(*info["moisture"]) + np.random.normal(0,3)
            N = np.clip(np.random.normal((info["N"][0]+info["N"][1])/2,5),0,200)
            P = np.clip(np.random.normal((info["P"][0]+info["P"][1])/2,4),0,200)
            K = np.clip(np.random.normal((info["K"][0]+info["K"][1])/2,6),0,200)
            soil = np.random.choice(SOIL_TYPES)
            rows.append({"Temparature":round(float(temp),1),"Humidity":round(float(hum),1),"Moisture":round(float(moist),1),
                         "Soil Type":soil,"Nitrogen":round(float(N),1),"Phosphorous":round(float(P),1),"Potassium":round(float(K),1),
                         "Fertilizer Name":fert})
    return pd.DataFrame(rows)

def train_fertilizer_model():
    df = load_fert_dataset()
    if "Fertilizer Name" not in df.columns:
        st.error("Dataset must have 'Fertilizer Name' column.")
        return
    features = [c for c in df.columns if c!="Fertilizer Name"]
    X = df[features].copy()
    encoders={}
    for c in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c]=le
    y_le = LabelEncoder(); y = y_le.fit_transform(df["Fertilizer Name"].astype(str))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.18, stratify=y, random_state=SEED)
    model = RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight='balanced', n_jobs=-1)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    rep = classification_report(yte, preds, zero_division=0)
    st.text("Fertilizer classifier report:")
    st.text(rep)
    joblib.dump(model, FERT_MODEL)
    joblib.dump(y_le, FERT_LE)
    joblib.dump(features, FERT_FEATS)
    joblib.dump(encoders, MODELS_DIR / "fert_encoders.joblib")
    defaults = {}
    for f in features:
        if X[f].dtype.kind in 'iufc':
            defaults[f]=float(X[f].median())
        else:
            defaults[f]=str(df[f].mode().iloc[0]) if not df[f].mode().empty else ""
    meta = {"features":features, "defaults":defaults, "clf_classes": model.classes_.tolist() if hasattr(model,"classes_") else list(y_le.classes_)}
    save_json(meta, FERT_META)
    st.success("Fertilizer model trained & saved.")

# ---------------- Prediction flow (fertilizer -> crops) ----------------
def load_fertifacts():
    if FERT_MODEL.exists() and FERT_LE.exists() and FERT_FEATS.exists() and FERT_META.exists():
        model = joblib.load(FERT_MODEL)
        le = joblib.load(FERT_LE)
        features = joblib.load(FERT_FEATS)
        meta = load_json(FERT_META)
        encs = joblib.load(MODELS_DIR / "fert_encoders.joblib") if (MODELS_DIR / "fert_encoders.joblib").exists() else {}
        return model, le, features, meta, encs
    return None, None, None, None, None

def predict_and_recommend(user_inputs):
    model, le, features, meta, encs = load_fertifacts()
    if model is None:
        st.error("No fertilizer model found. Train first.")
        return None
    defaults = meta.get("defaults", {})
    mapped, missing = map_features(features, user_inputs, defaults)
    if missing:
        st.warning(f"Missing inputs filled with defaults: {missing}")
    row = []
    for f in features:
        v = mapped.get(f, 0)
        if f in encs:
            enc = encs[f]
            try:
                v_enc = enc.transform([str(v)])[0]
                row.append(float(v_enc))
            except:
                try:
                    row.append(float(enc.transform([enc.classes_[0]])[0]))
                except:
                    row.append(0.0)
        else:
            try:
                row.append(float(v))
            except:
                row.append(0.0)
    Xv = np.array(row).reshape(1,-1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xv)[0]
        try:
            classes = le.inverse_transform(np.arange(len(probs)))
        except:
            classes = [str(i) for i in range(len(probs))]
        prob_dict = {classes[i]: float(probs[i]) for i in range(len(probs))}
        idx = int(probs.argmax())
    else:
        idx = int(model.predict(Xv)[0])
        prob_dict = {}
    fert_label = le.inverse_transform([idx])[0]
    crop_list = recommend_crops_from_fert_and_env(fert_label, user_inputs)
    return fert_label, prob_dict, crop_list, mapped

# ---------------- Price: features, training, prediction ----------------
def make_price_features(df):
    df = df.copy()
    # normalize date
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col).reset_index(drop=True)
        df = df.rename(columns={date_col:'date'})
    else:
        # create a date if none exists
        df['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    # detect price column
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'price' in df.columns:
        price_col = 'price'
    elif numcols:
        price_col = numcols[-1]
        df = df.rename(columns={price_col:'price'})
        price_col = 'price'
    else:
        raise ValueError("No numeric price column found in price data")
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price']).reset_index(drop=True)
    df['lag1'] = df['price'].shift(1)
    df['lag7'] = df['price'].shift(7)
    df['lag30'] = df['price'].shift(30)
    df['rolling7'] = df['price'].rolling(7, min_periods=1).mean()
    df['rolling30'] = df['price'].rolling(30, min_periods=1).mean()
    df['vol30'] = df['price'].rolling(30, min_periods=1).std().fillna(0.0)
    df = df.dropna().reset_index(drop=True)
    return df

def train_price_models(df_price, time_splits=5, min_rows_per_crop=60):
    st.info("Training price models (this may take time)...")
    df = df_price.copy()
    trained = {}
    # if 'crop' present, train per-crop models
    if 'crop' in df.columns:
        for crop, sub in df.groupby('crop'):
            if len(sub) < min_rows_per_crop:
                st.write(f"Skipping {crop}: insufficient rows ({len(sub)})")
                continue
            try:
                subf = make_price_features(sub)
            except Exception as e:
                st.write(f"Failed to make features for {crop}: {e}")
                continue
            X = subf[['lag1','lag7','lag30','rolling7','rolling30','vol30']].values
            y = subf['price'].values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            joblib.dump(scaler, Path(PRICE_MODEL_DIR)/f"scaler_{crop}.joblib")
            tscv = TimeSeriesSplit(n_splits=min(time_splits, max(2, len(Xs)//10)))
            maes = {10:[],50:[],90:[]}
            for q,alpha in [(10,0.1),(50,0.5),(90,0.9)]:
                mdl = GradientBoostingRegressor(loss='quantile', alpha=alpha, n_estimators=200, random_state=SEED)
                # do simple CV
                for train_idx, test_idx in tscv.split(Xs):
                    mdl.fit(Xs[train_idx], y[train_idx])
                    preds = mdl.predict(Xs[test_idx]); maes[q].append(mean_absolute_error(y[test_idx], preds))
                mdl.fit(Xs, y)
                joblib.dump(mdl, Path(PRICE_MODEL_DIR)/f"price_{crop}_q{q}.joblib")
            trained[crop] = {q: float(np.mean(maes[q])) for q in maes}
            st.write(f"Trained {crop} (CV MAEs): {trained[crop]}")
    # if nothing trained, train generic on full df
    if not trained:
        try:
            subf = make_price_features(df)
            X = subf[['lag1','lag7','lag30','rolling7','rolling30','vol30']].values
            y = subf['price'].values
            scaler = StandardScaler(); Xs = scaler.fit_transform(X)
            joblib.dump(scaler, Path(PRICE_MODEL_DIR)/"scaler_generic.joblib")
            tscv = TimeSeriesSplit(n_splits=min(time_splits, max(2, len(Xs)//10)))
            maes = {}
            for q,alpha in [(10,0.1),(50,0.5),(90,0.9)]:
                mdl = GradientBoostingRegressor(loss='quantile', alpha=alpha, n_estimators=200, random_state=SEED)
                fold_maes=[]
                for train_idx,test_idx in tscv.split(Xs):
                    mdl.fit(Xs[train_idx], y[train_idx])
                    preds = mdl.predict(Xs[test_idx]); fold_maes.append(mean_absolute_error(y[test_idx], preds))
                mdl.fit(Xs,y)
                joblib.dump(mdl, Path(PRICE_MODEL_DIR)/f"price_generic_q{q}.joblib")
                maes[q] = float(np.mean(fold_maes))
            trained['generic'] = maes
            st.write("Trained generic price models (CV MAEs):", maes)
        except Exception as e:
            st.error("Failed to train generic price models: " + str(e))
    st.success("Price model training complete.")
    return trained

def compute_recent_rolling_features(crop_name, df_price=None, min_history=30):
    """
    If df_price provided, filter by crop column if present and compute recent lags/rolling.
    Otherwise load PRICE_FILE.
    """
    if df_price is None:
        if PRICE_FILE.exists():
            try:
                df_price = pd.read_csv(PRICE_FILE, low_memory=False)
            except Exception:
                return None
        else:
            return None
    # normalize
    date_col = next((c for c in df_price.columns if 'date' in c.lower()), None)
    if date_col:
        df_price[date_col] = pd.to_datetime(df_price[date_col], errors='coerce')
        df_price = df_price.sort_values(date_col).reset_index(drop=True)
    else:
        df_price['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_price))
    # filter by crop if column exists
    if 'crop' in df_price.columns:
        sub = df_price[df_price['crop'].astype(str).str.lower()==crop_name.lower()].copy()
    else:
        sub = df_price.copy()
    if sub.empty: return None
    price_col = sub.select_dtypes(include=[np.number]).columns[-1]
    series = sub[price_col].dropna().reset_index(drop=True)
    if series.empty: return None
    recent = series.tail(min_history)
    lag1 = float(series.iloc[-1])
    lag7 = float(series.iloc[-7]) if len(series)>=7 else float(series.iloc[0])
    lag30 = float(series.iloc[-30]) if len(series)>=30 else float(series.iloc[0])
    rolling7 = float(recent.rolling(7, min_periods=1).mean().iloc[-1])
    rolling30 = float(recent.rolling(30, min_periods=1).mean().iloc[-1])
    vol30 = float(recent.rolling(30, min_periods=1).std().iloc[-1]) if len(recent)>1 else 0.0
    return {'lag1':lag1,'lag7':lag7,'lag30':lag30,'rolling7':rolling7,'rolling30':rolling30,'vol30':vol30}

def predict_price_for_crop(chosen_crop, l1=None, l7=None, l30=None, df_price=None):
    """
    Use per-crop models if exist else generic.
    l1/l7/l30 optional — if None and df_price provided compute from history.
    Returns dict {10:val,50:val,90:val}
    """
    # try compute recent features
    recent = None
    if df_price is not None:
        try:
            recent = compute_recent_rolling_features(chosen_crop, df_price=df_price)
        except Exception:
            recent = None
    else:
        try:
            recent = compute_recent_rolling_features(chosen_crop)
        except Exception:
            recent = None
    if recent:
        l1 = recent['lag1'] if (l1 is None or l1==0) else l1
        l7 = recent['lag7'] if (l7 is None or l7==0) else l7
        l30 = recent['lag30'] if (l30 is None or l30==0) else l30
        rolling7 = recent['rolling7']; rolling30 = recent['rolling30']; vol30 = recent['vol30']
    else:
        rolling7 = l7 if (l7 and l7!=0) else (l1 or 0.0)
        rolling30 = l30 if (l30 and l30!=0) else (l1 or 0.0)
        vol30 = 0.0
    preds = {10:None,50:None,90:None}
    scaler_path = Path(PRICE_MODEL_DIR)/f"scaler_{chosen_crop}.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        for q in (10,50,90):
            mpath = Path(PRICE_MODEL_DIR)/f"price_{chosen_crop}_q{q}.joblib"
            if mpath.exists():
                mdl = joblib.load(mpath)
                feat = [l1 or 0.0, l7 or 0.0, l30 or 0.0, rolling7, rolling30, vol30]
                Xs = scaler.transform([feat])
                preds[q] = float(mdl.predict(Xs)[0])
        return preds
    # generic
    gen_scaler = Path(PRICE_MODEL_DIR)/"scaler_generic.joblib"
    if gen_scaler.exists():
        scaler = joblib.load(gen_scaler)
        for q in (10,50,90):
            mpath = Path(PRICE_MODEL_DIR)/f"price_generic_q{q}.joblib"
            if mpath.exists():
                mdl = joblib.load(mpath)
                feat = [l1 or 0.0, l7 or 0.0, l30 or 0.0, rolling7, rolling30, vol30]
                Xs = scaler.transform([feat])
                preds[q] = float(mdl.predict(Xs)[0])
        return preds
    return preds

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Krishi Sahayak", layout="wide")
st.title("Krishi Sahayak — Crop Suggestion & Price Forecast")

with st.sidebar:
    st.header("Model Controls")
    if st.button("Train Fertilizer Model"):
        train_fertilizer_model()
    st.markdown("---")
    st.subheader("Price model")
    st.write("Upload a price CSV below or train on existing data/Crops_data.csv")
    uploaded_price = st.file_uploader("Upload price CSV (columns: date, crop(optional), price)", type=["csv"])
    if st.button("Train Price Models (using uploaded CSV if present)"):
        try:
            if uploaded_price is not None:
                df_price = pd.read_csv(uploaded_price)
            elif PRICE_FILE.exists():
                df_price = pd.read_csv(PRICE_FILE, low_memory=False)
            else:
                st.error("No price CSV found to train on. Upload one or place data/Crops_data.csv")
                df_price = None
            if df_price is not None:
                res = train_price_models(df_price)
                st.write("Price training result summary:", res)
        except Exception as e:
            st.error("Price training failed: " + str(e))
    st.markdown("---")
    st.write("Model artifacts folder: " + str(MODELS_DIR))
    if st.button("Show prediction log"):
        if PRED_LOG.exists():
            st.write(pd.read_csv(PRED_LOG).tail(40))
        else:
            st.info("No prediction log yet.")

# User Inputs (map to features in your data_core.csv)
st.subheader("Field Inputs for crop suggestion")
c1,c2,c3 = st.columns(3)
user_inputs = {}
user_inputs["Temparature"] = c1.number_input("Temperature (°C)", 5.0, 45.0, 25.0)
user_inputs["Humidity"] = c2.number_input("Humidity (%)", 10.0, 100.0, 60.0)
user_inputs["Moisture"] = c3.number_input("Moisture (%)", 0.0, 100.0, 40.0)
user_inputs["Soil Type"] = st.selectbox("Soil Type", SOIL_TYPES, index=1)
user_inputs["Nitrogen"] = st.number_input("Nitrogen (N)", 0.0, 200.0, 30.0)
user_inputs["Phosphorous"] = st.number_input("Phosphorous (P)", 0.0, 200.0, 20.0)
user_inputs["Potassium"] = st.number_input("Potassium (K)", 0.0, 200.0, 20.0)

st.markdown("---")
st.subheader("Price Forecast Inputs")
st.write("You can either upload a historical price CSV for the chosen crop (preferred) or provide last-day / 7-day / 30-day prices manually.")
price_upload = st.file_uploader("Upload historical price CSV (optional) — columns: date,crop,price", type=["csv"], key="price_upload2")
chosen_crop_for_price = st.text_input("Crop name for price forecast (exact match to CSV 'crop' column)", value="")
l1 = st.number_input("Last price (1 day ago) — leave 0 to use uploaded history", value=0.0, step=1.0)
l7 = st.number_input("Price 7 days ago — leave 0 to use uploaded history", value=0.0, step=1.0)
l30 = st.number_input("Price 30 days ago — leave 0 to use uploaded history", value=0.0, step=1.0)

if st.button("Predict & Recommend Crops + Price"):
    # fertilizer -> crop
    model, le, features, meta, encs = load_fertifacts()
    if model is None:
        st.error("No trained fertilizer model found. Please train it from the sidebar.")
    else:
        feat_list = meta['features']
        defaults = meta.get('defaults', {})
        mapped, missing = map_features(feat_list, user_inputs, defaults)
        if missing:
            st.warning("Missing fields filled using training defaults: " + ", ".join(missing))
        # build input vector
        row=[]
        encoders = joblib.load(MODELS_DIR / "fert_encoders.joblib") if (MODELS_DIR / "fert_encoders.joblib").exists() else {}
        for f in feat_list:
            v = mapped.get(f, 0)
            if f in encoders:
                enc = encoders[f]
                try:
                    v_enc = enc.transform([str(v)])[0]; row.append(float(v_enc))
                except:
                    try:
                        row.append(float(enc.transform([enc.classes_[0]])[0]))
                    except:
                        row.append(0.0)
            else:
                try:
                    row.append(float(v))
                except:
                    row.append(0.0)
        Xv = np.array(row).reshape(1,-1)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xv)[0]
            classes = le.inverse_transform(np.arange(len(probs)))
            prob_dict = {classes[i]: float(probs[i]) for i in range(len(probs))}
            idx = int(probs.argmax())
        else:
            idx = int(model.predict(Xv)[0]); prob_dict={}
        fert_label = le.inverse_transform([idx])[0]
        st.success(f"Predicted fertilizer: {fert_label}")
        st.subheader("Fertilizer probabilities")
        fig = go.Figure([go.Bar(x=list(prob_dict.keys()), y=list(prob_dict.values()))])
        fig.update_layout(yaxis_title="Probability", height=320)
        st.plotly_chart(fig, use_container_width=True)
        crop_candidates = recommend_crops_from_fert_and_env(fert_label, user_inputs, top_k=6)
        st.subheader("Recommended crops (ranked by suitability)")
        for c in crop_candidates:
            sc = crop_suitability_score(c, user_inputs)
            st.write(f"- {c} (suitability: {sc:.2f})")

        # Price forecasting
        df_price = None
        if price_upload is not None:
            try:
                df_price = pd.read_csv(price_upload)
            except Exception as e:
                st.warning("Failed to read uploaded price CSV: " + str(e))
                df_price = None
        elif PRICE_FILE.exists():
            try:
                df_price = pd.read_csv(PRICE_FILE, low_memory=False)
            except Exception:
                df_price = None

        chosen_crop = chosen_crop_for_price.strip() if chosen_crop_for_price.strip() else (crop_candidates[0] if crop_candidates else None)
        if chosen_crop is None:
            st.warning("No crop chosen for price forecasting.")
        else:
            st.info(f"Forecasting price for: {chosen_crop}")
            # show historical series if available
            if df_price is not None:
                # try to plot crop-specific history
                if 'crop' in df_price.columns:
                    hist = df_price[df_price['crop'].astype(str).str.lower()==chosen_crop.lower()].copy()
                else:
                    hist = df_price.copy()
                if not hist.empty:
                    # unify date/price columns
                    date_col = next((c for c in hist.columns if 'date' in c.lower()), None)
                    if date_col:
                        hist[date_col] = pd.to_datetime(hist[date_col], errors='coerce')
                        hist = hist.sort_values(date_col)
                        price_col = next((c for c in hist.columns if hist[c].dtype.kind in 'if' and c.lower()!='year'), None)
                        if price_col is None:
                            price_col = [c for c in hist.columns if c not in (['date','crop']) and hist[c].dtype.kind in 'if']
                            price_col = price_col[-1] if price_col else None
                        if price_col:
                            fig2 = go.Figure([go.Scatter(x=hist[date_col], y=hist[price_col], mode='lines', name='price')])
                            fig2.update_layout(title=f"Historical prices for {chosen_crop}", xaxis_title="Date", yaxis_title="Price", height=300)
                            st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No historical series for this crop in uploaded file.")
            # Predict using models (train if models not present)
            # if price models missing, prompt to train in sidebar
            preds = predict_price_for_crop(chosen_crop, l1 if l1>0 else None, l7 if l7>0 else None, l30 if l30>0 else None, df_price)
            st.markdown("### Price scenarios")
            labels = ["Low (10%)","Median (50%)","High (90%)"]
            vals = [preds.get(10), preds.get(50), preds.get(90)]
            fig3 = go.Figure([go.Bar(x=labels, y=vals)])
            fig3.update_layout(yaxis_title="Estimated Price", height=320)
            st.plotly_chart(fig3, use_container_width=True)

        # log
        log = {"timestamp": now_str(), "fertilizer": fert_label, "chosen_crop_for_price": chosen_crop}
        for k,v in mapped.items(): log[f"input_{k}"] = v
        log["recommended_crops"] = ";".join(crop_candidates)
        if 'preds' in locals():
            log["pred_low"] = preds.get(10); log["pred_med"] = preds.get(50); log["pred_high"] = preds.get(90)
        append_log(log)
        st.info("Prediction logged.")

st.markdown("---")
st.write("© Krishi Sahayak —(2025)")
