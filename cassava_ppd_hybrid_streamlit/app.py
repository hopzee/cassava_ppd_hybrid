from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Cassava PPD Risk Predictor", page_icon="🌱", layout="wide")

SEQ_LEN = 24
HORIZON = 72
ENV_COLS = ["temperature_c", "relative_humidity_pct"]
REQUIRED_ENV = {"batch_id", "timestamp", *ENV_COLS}
REQUIRED_ROOT = {
    "batch_id", "variety", "mechanical_damage", "initial_condition",
    "dry_matter_pct", "ppd_status_72h", "harvest_timestamp",
}

st.title("🌱 Cassava Post-Harvest Deterioration Risk Predictor")
st.caption("Bi-LSTM environmental forecasting + XGBoost cassava deterioration-risk estimation")
st.info(
    "Research prototype. It requires genuinely linked environmental records and independently observed "
    "cassava-root outcomes for research evaluation. The built-in synthetic data are for software testing only."
)


def make_demo_data(seed: int = 21, batches: int = 40) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic records solely to test the application workflow."""
    rng = np.random.default_rng(seed)
    env_rows, root_rows = [], []
    base_start = pd.Timestamp("2025-01-01 06:00:00")
    for b in range(batches):
        batch_id = f"DEMO-{b + 1:03d}"
        harvest_time = base_start + pd.Timedelta(days=b * 4)
        variety = rng.choice(["TME 419", "TMS 30572", "Local sweet", "Local bitter"])
        damage = int(rng.integers(0, 4))
        initial = rng.choice(["Good", "Fair", "Poor"], p=[0.55, 0.30, 0.15])
        dry = float(np.clip(rng.normal(34, 3), 20, 45))
        base_t = float(rng.uniform(24, 31))
        base_rh = float(rng.uniform(55, 82))
        future_values = []
        for h in range(-SEQ_LEN, HORIZON):
            temp = base_t + 1.8 * np.sin(h / 9) + rng.normal(0, 0.8)
            rh = base_rh + 5 * np.sin(h / 13 + 0.8) + rng.normal(0, 2.0)
            rh = float(np.clip(rh, 25, 98))
            env_rows.append({
                "batch_id": batch_id,
                "timestamp": harvest_time + pd.Timedelta(hours=h),
                "temperature_c": round(temp, 2),
                "relative_humidity_pct": round(rh, 2),
            })
            if h >= 0:
                future_values.append((temp, rh))
        future_t = float(np.mean([x[0] for x in future_values]))
        future_rh = float(np.mean([x[1] for x in future_values]))
        # Simulated labels are deliberately not real cassava observations.
        latent = (0.10 * (future_t - 27) + 0.018 * (future_rh - 70)
                  + 0.35 * damage
                  + (0.50 if initial == "Poor" else 0.20 if initial == "Fair" else 0)
                  + rng.normal(0, 0.45))
        root_rows.append({
            "batch_id": batch_id, "variety": variety,
            "mechanical_damage": damage, "initial_condition": initial,
            "dry_matter_pct": round(dry, 2), "ppd_status_72h": int(latent > 0.55),
            "harvest_timestamp": harvest_time,
        })
    return pd.DataFrame(env_rows), pd.DataFrame(root_rows)


def normalize_inputs(env: pd.DataFrame, roots: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    env, roots = env.copy(), roots.copy()
    env.columns = [str(c).strip().lower() for c in env.columns]
    roots.columns = [str(c).strip().lower() for c in roots.columns]
    missing_e, missing_r = REQUIRED_ENV - set(env.columns), REQUIRED_ROOT - set(roots.columns)
    if missing_e or missing_r:
        raise ValueError(
            f"Missing columns. Environmental file: {sorted(missing_e)}. "
            f"Cassava outcome file: {sorted(missing_r)}."
        )
    env["batch_id"] = env["batch_id"].astype(str).str.strip()
    roots["batch_id"] = roots["batch_id"].astype(str).str.strip()
    env["timestamp"] = pd.to_datetime(env["timestamp"], errors="coerce")
    for col in ENV_COLS:
        env[col] = pd.to_numeric(env[col], errors="coerce")
    for col in ["mechanical_damage", "dry_matter_pct", "ppd_status_72h"]:
        roots[col] = pd.to_numeric(roots[col], errors="coerce")
    roots["harvest_timestamp"] = pd.to_datetime(roots["harvest_timestamp"], errors="coerce")
    env = env.dropna(subset=["batch_id", "timestamp"] + ENV_COLS)
    roots = roots.dropna(subset=["batch_id", "mechanical_damage", "dry_matter_pct", "ppd_status_72h", "harvest_timestamp"])
    env = env.sort_values(["batch_id", "timestamp"]).drop_duplicates(["batch_id", "timestamp"])
    roots = roots.drop_duplicates("batch_id", keep="last")
    if not roots["ppd_status_72h"].isin([0, 1]).all():
        raise ValueError("ppd_status_72h must be binary: 0 = no defined event; 1 = defined PPD event.")
    roots["ppd_status_72h"] = roots["ppd_status_72h"].astype(int)
    if not roots["mechanical_damage"].between(0, 3).all():
        raise ValueError("mechanical_damage must use the documented 0–3 scale.")
    if not roots["dry_matter_pct"].between(0, 100).all():
        raise ValueError("dry_matter_pct must be between 0 and 100.")
    if roots["variety"].isna().any() or roots["initial_condition"].isna().any():
        raise ValueError("variety and initial_condition cannot be blank.")
    matched = set(env.batch_id) & set(roots.batch_id)
    if len(matched) < 20:
        raise ValueError(f"At least 20 linked batches are recommended for a 70/15/15 split; found {len(matched)}.")
    return env[env.batch_id.isin(matched)].copy(), roots[roots.batch_id.isin(matched)].copy()


def build_batch_windows(env: pd.DataFrame, roots: pd.DataFrame):
    """Build aligned pre-harvest 24h input and post-harvest 72h target windows."""
    root_map = roots.set_index("batch_id").to_dict("index")
    samples = []
    for batch_id, group in env.groupby("batch_id"):
        if batch_id not in root_map:
            continue
        group = group.sort_values("timestamp")
        harvest = pd.Timestamp(root_map[batch_id]["harvest_timestamp"])
        history_start, forecast_end = harvest - pd.Timedelta(hours=SEQ_LEN), harvest + pd.Timedelta(hours=HORIZON)
        history = group[(group.timestamp >= history_start) & (group.timestamp < harvest)]
        future = group[(group.timestamp >= harvest) & (group.timestamp < forecast_end)]
        # Exact hourly coverage is required so sequence positions have a consistent meaning.
        if len(history) != SEQ_LEN or len(future) != HORIZON:
            continue
        if history.timestamp.duplicated().any() or future.timestamp.duplicated().any():
            continue
        if not np.all(np.diff(history.timestamp.astype("int64")) == pd.Timedelta(hours=1).value):
            continue
        if not np.all(np.diff(future.timestamp.astype("int64")) == pd.Timedelta(hours=1).value):
            continue
        x = history[ENV_COLS].to_numpy(dtype=np.float32)
        y = future[ENV_COLS].to_numpy(dtype=np.float32)
        samples.append({
            "batch_id": batch_id, "x": x, "y": y,
            "root": root_map[batch_id], "first_timestamp": harvest,
        })
    if len(samples) < 20:
        raise ValueError(
            f"Only {len(samples)} batches have complete aligned records. Each batch needs 24 hourly "
            "environmental observations immediately before harvest and 72 hourly observations after harvest. "
            "At least 20 batches are recommended."
        )
    return samples


def chronological_group_split(samples):
    """Split whole batches by their first timestamp to avoid batch/window leakage."""
    ordered = sorted(samples, key=lambda item: (item["first_timestamp"], item["batch_id"]))
    n = len(ordered)
    n_train = int(np.floor(n * 0.70))
    n_val = int(np.floor(n * 0.15))
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 2:
        raise ValueError("Not enough batches for a meaningful chronological 70/15/15 split.")
    return ordered[:n_train], ordered[n_train:n_train+n_val], ordered[n_train+n_val:]


def forecast_summary(pred: np.ndarray) -> Dict[str, float]:
    temp, rh = pred[:, 0], pred[:, 1]
    return {
        "forecast_temp_mean": float(np.mean(temp)),
        "forecast_temp_max": float(np.max(temp)),
        "forecast_temp_min": float(np.min(temp)),
        "forecast_rh_mean": float(np.mean(rh)),
        "forecast_rh_max": float(np.max(rh)),
        "forecast_rh_min": float(np.min(rh)),
        "temp_hours_above_30c": float(np.sum(temp > 30)),
        "rh_hours_above_80pct": float(np.sum(rh > 80)),
    }


def root_feature_row(root: dict) -> dict:
    harvest_hour = 0.0
    if pd.notna(root.get("harvest_timestamp")):
        harvest_hour = float(pd.Timestamp(root["harvest_timestamp"]).hour)
    return {
        "mechanical_damage": float(root["mechanical_damage"]),
        "dry_matter_pct": float(root["dry_matter_pct"]),
        "variety": str(root["variety"]),
        "initial_condition": str(root["initial_condition"]),
        "harvest_hour": harvest_hour,
    }


def build_bilstm():
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Reshape
        model = Sequential([
            Input(shape=(SEQ_LEN, len(ENV_COLS))),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.15),
            Dense(128, activation="relu"),
            Dense(HORIZON * len(ENV_COLS)),
            Reshape((HORIZON, len(ENV_COLS))),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow could not be loaded. Install requirements.txt using Python 3.11, then restart. "
            f"Details: {exc}"
        )


def make_feature_frame(records, forecasts):
    rows = []
    for record, forecast in zip(records, forecasts):
        rows.append({
            "batch_id": record["batch_id"],
            **forecast_summary(forecast),
            **root_feature_row(record["root"]),
            "target": int(record["root"]["ppd_status_72h"]),
        })
    return pd.DataFrame(rows)


def encode_features(train_df, other_df=None):
    categorical = ["variety", "initial_condition"]
    train_x = pd.get_dummies(train_df.drop(columns=["batch_id", "target"]), columns=categorical)
    if other_df is None:
        return train_x, list(train_x.columns)
    other_x = pd.get_dummies(other_df.drop(columns=["batch_id", "target"]), columns=categorical)
    return train_x, other_x.reindex(columns=train_x.columns, fill_value=0), list(train_x.columns)


def score_classifier(model, x, y):
    probabilities = model.predict_proba(x)[:, 1]
    labels = (probabilities >= 0.5).astype(int)
    scores = {
        "Accuracy": float(accuracy_score(y, labels)),
        "Precision": float(precision_score(y, labels, zero_division=0)),
        "Recall": float(recall_score(y, labels, zero_division=0)),
        "F1-score": float(f1_score(y, labels, zero_division=0)),
    }
    if len(np.unique(y)) == 2:
        scores["ROC-AUC"] = float(roc_auc_score(y, probabilities))
    return scores, probabilities, labels


def train_pipeline(env, roots, epochs=15):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    samples = build_batch_windows(env, roots)
    train, validation, test = chronological_group_split(samples)
    X_train = np.stack([s["x"] for s in train])
    Y_train = np.stack([s["y"] for s in train])
    X_val = np.stack([s["x"] for s in validation])
    Y_val = np.stack([s["y"] for s in validation])
    X_test = np.stack([s["x"] for s in test])
    Y_test = np.stack([s["y"] for s in test])

    # Fit all scaling parameters on training batches only.
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaler.fit(X_train.reshape(-1, len(ENV_COLS)))
    y_scaler.fit(Y_train.reshape(-1, len(ENV_COLS)))

    def scale_x(x):
        return x_scaler.transform(x.reshape(-1, len(ENV_COLS))).reshape(-1, SEQ_LEN, len(ENV_COLS))

    def scale_y(y):
        return y_scaler.transform(y.reshape(-1, len(ENV_COLS))).reshape(-1, HORIZON, len(ENV_COLS))

    model = build_bilstm()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)]
    history = model.fit(
        scale_x(X_train), scale_y(Y_train),
        validation_data=(scale_x(X_val), scale_y(Y_val)),
        epochs=int(epochs), batch_size=min(16, len(train)), verbose=0, callbacks=callbacks,
    )

    def predict_environment(x):
        scaled = model.predict(scale_x(x), verbose=0)
        return y_scaler.inverse_transform(scaled.reshape(-1, len(ENV_COLS))).reshape(-1, HORIZON, len(ENV_COLS))

    pred_test = predict_environment(X_test)
    forecast_metrics = {}
    for idx, name in enumerate(["Temperature", "Relative humidity"]):
        actual = Y_test[:, :, idx].ravel()
        predicted = pred_test[:, :, idx].ravel()
        forecast_metrics[f"Bi-LSTM {name} MAE"] = float(mean_absolute_error(actual, predicted))
        forecast_metrics[f"Bi-LSTM {name} RMSE"] = float(np.sqrt(mean_squared_error(actual, predicted)))
        # Persistence baseline: repeat the last observed value for all 72 future hours.
        persistence = np.repeat(X_test[:, -1:, idx], HORIZON, axis=1).ravel()
        forecast_metrics[f"Persistence {name} MAE"] = float(mean_absolute_error(actual, persistence))
        forecast_metrics[f"Persistence {name} RMSE"] = float(np.sqrt(mean_squared_error(actual, persistence)))

    train_forecasts = model.predict(scale_x(X_train), verbose=0)
    val_forecasts = model.predict(scale_x(X_val), verbose=0)
    test_forecasts = model.predict(scale_x(X_test), verbose=0)
    def inverse_batch(pred):
        return y_scaler.inverse_transform(pred.reshape(-1, len(ENV_COLS))).reshape(-1, HORIZON, len(ENV_COLS))
    train_records = [{"batch_id": s["batch_id"], "root": s["root"]} for s in train]
    val_records = [{"batch_id": s["batch_id"], "root": s["root"]} for s in validation]
    test_records = [{"batch_id": s["batch_id"], "root": s["root"]} for s in test]
    train_features = make_feature_frame(train_records, inverse_batch(train_forecasts))
    val_features = make_feature_frame(val_records, inverse_batch(val_forecasts))
    test_features = make_feature_frame(test_records, inverse_batch(test_forecasts))

    if train_features.target.nunique() < 2:
        raise ValueError("The training batches contain only one outcome class. Add labelled batches from both classes.")
    Xtr, Xval, cols = encode_features(train_features, val_features)
    _, Xte, _ = encode_features(train_features, test_features)
    ytr = train_features.target.astype(int)
    yval = val_features.target.astype(int)
    yte = test_features.target.astype(int)

    # The chapter's specified main risk model is XGBoost; Random Forest is a comparator.
    xgb = XGBClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=2,
    )
    xgb.fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=2)
    rf.fit(Xtr, ytr)
    validation_scores = {
        "XGBoost": score_classifier(xgb, Xval, yval)[0] if len(np.unique(yval)) > 1 else {"F1-score": float(f1_score(yval, xgb.predict(Xval), zero_division=0))},
        "Random Forest": score_classifier(rf, Xval, yval)[0] if len(np.unique(yval)) > 1 else {"F1-score": float(f1_score(yval, rf.predict(Xval), zero_division=0))},
    }
    xgb_test, xgb_prob, xgb_label = score_classifier(xgb, Xte, yte)
    rf_test = score_classifier(rf, Xte, yte)[0]
    return {
        "forecast_model": model, "x_scaler": x_scaler, "y_scaler": y_scaler,
        "risk_model": xgb, "rf_model": rf, "feature_columns": cols,
        "forecast_metrics": forecast_metrics, "risk_metrics": xgb_test,
        "rf_metrics": rf_test, "validation_metrics": validation_scores,
        "history": history.history,
        "test_confusion": confusion_matrix(yte, xgb_label, labels=[0, 1]).tolist(),
        "train_batches": len(train), "validation_batches": len(validation),
        "test_batches": len(test), "risk_test_rows": len(test_features),
        "training_batch_ids": [s["batch_id"] for s in train],
        "validation_batch_ids": [s["batch_id"] for s in validation],
        "test_batch_ids": [s["batch_id"] for s in test],
    }


def predict_batch(bundle, history_df, variety, damage, initial, dry, harvest_hour):
    history_df = history_df.sort_values("timestamp")
    if len(history_df) < SEQ_LEN:
        raise ValueError(f"At least {SEQ_LEN} recent hourly observations are required; found {len(history_df)}.")
    seq = history_df[ENV_COLS].tail(SEQ_LEN).to_numpy(dtype=float)
    x = bundle["x_scaler"].transform(seq).reshape(1, SEQ_LEN, len(ENV_COLS))
    scaled = bundle["forecast_model"].predict(x, verbose=0)[0]
    forecast = bundle["y_scaler"].inverse_transform(scaled.reshape(-1, len(ENV_COLS))).reshape(HORIZON, len(ENV_COLS))
    features = {
        **forecast_summary(forecast), "mechanical_damage": float(damage),
        "dry_matter_pct": float(dry), "harvest_hour": float(harvest_hour),
        "variety": str(variety), "initial_condition": str(initial),
    }
    frame = pd.get_dummies(pd.DataFrame([features]), columns=["variety", "initial_condition"])
    frame = frame.reindex(columns=bundle["feature_columns"], fill_value=0)
    probability = float(bundle["risk_model"].predict_proba(frame)[0, 1])
    return forecast, probability


with st.sidebar:
    st.header("Data and model")
    data_mode = st.radio("Data source", ["Use supplied CSV files", "Load synthetic demo data"])
    env_file = st.file_uploader("Environmental time-series CSV", type=["csv"], disabled=data_mode != "Use supplied CSV files")
    root_file = st.file_uploader("Cassava phenotype/outcome CSV", type=["csv"], disabled=data_mode != "Use supplied CSV files")
    epochs = st.slider("Bi-LSTM training epochs", 5, 40, 15)
    train_btn = st.button("Train hybrid framework", type="primary", disabled=(data_mode == "Use supplied CSV files" and (env_file is None or root_file is None)))

if data_mode == "Load synthetic demo data":
    env_df, root_df = make_demo_data()
    st.warning("Synthetic demonstration mode is active. Model metrics from this mode are not real-world cassava PPD validation.")
else:
    env_df, root_df = None, None
    if env_file is not None and root_file is not None:
        try:
            env_df, root_df = pd.read_csv(env_file), pd.read_csv(root_file)
        except Exception as exc:
            st.error(f"Could not read uploaded CSV files: {exc}")

if env_df is not None and root_df is not None:
    try:
        env_df, root_df = normalize_inputs(env_df, root_df)
        st.success(f"Loaded {env_df.batch_id.nunique()} linked batches and {len(root_df)} cassava outcome records.")
        with st.expander("Preview environmental records"):
            st.dataframe(env_df.head(20), use_container_width=True)
        with st.expander("Preview cassava batch records"):
            st.dataframe(root_df.head(20), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))
        env_df, root_df = None, None

if "bundle" not in st.session_state:
    st.session_state.bundle = None
if train_btn:
    if env_df is None or root_df is None:
        st.error("Load valid environmental and cassava CSV files first.")
    else:
        try:
            with st.spinner("Training the Bi-LSTM forecaster and XGBoost risk model. This may take several minutes..."):
                st.session_state.bundle = train_pipeline(env_df, root_df, epochs)
                st.session_state.training_mode = data_mode
                st.session_state.training_env = env_df.copy()
                st.session_state.training_roots = root_df.copy()
            st.success("Hybrid model training completed.")
        except Exception as exc:
            st.error(f"Training could not be completed: {exc}")

bundle = st.session_state.bundle
if bundle:
    st.divider()
    st.header("Model evaluation")
    st.caption(
        f"Chronological batch split — training: {bundle['train_batches']} (70%); "
        f"validation: {bundle['validation_batches']} (15%); test: {bundle['test_batches']} (15%)."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Environmental forecasting")
        st.dataframe(pd.DataFrame([bundle["forecast_metrics"]]).T.rename(columns={0: "Value"}), use_container_width=True)
    with col2:
        st.subheader("XGBoost deterioration-risk metrics (test set)")
        st.dataframe(pd.DataFrame([bundle["risk_metrics"]]).T.rename(columns={0: "Value"}), use_container_width=True)
    st.subheader("Risk model baseline comparison (held-out test set)")
    st.dataframe(pd.DataFrame({"XGBoost": bundle["risk_metrics"], "Random Forest": bundle["rf_metrics"]}).fillna("Not defined"), use_container_width=True)
    st.subheader("Validation-set model comparison")
    st.dataframe(pd.DataFrame(bundle["validation_metrics"]).fillna("Not defined"), use_container_width=True)
    st.write("XGBoost held-out confusion matrix (rows = actual; columns = predicted)")
    st.dataframe(pd.DataFrame(bundle["test_confusion"], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]), use_container_width=True)
    if st.session_state.get("training_mode") == "Load synthetic demo data":
        st.warning("All displayed metrics are generated from synthetic records. They must not be reported as measured performance on real cassava roots.")

    st.divider()
    st.header("Estimate risk for a cassava batch")
    st.caption("Choose an existing batch from your data or enter a new batch manually. Both options use the same trained Bi-LSTM and XGBoost prediction pipeline.")
    prediction_mode = st.radio(
        "Prediction input method",
        ["Use a batch from the loaded dataset", "Enter batch features manually"],
        horizontal=True,
        key="prediction_input_mode",
    )

    if prediction_mode == "Use a batch from the loaded dataset":
        trained_env = st.session_state.training_env
        trained_roots = st.session_state.training_roots
        selected_batch = st.selectbox("Select a batch from the loaded dataset", sorted(trained_env.batch_id.unique()))
        root_rows = trained_roots[trained_roots.batch_id == selected_batch]
        harvest_time = pd.Timestamp(root_rows.iloc[0]["harvest_timestamp"]) if len(root_rows) else pd.NaT
        recent = trained_env[(trained_env.batch_id == selected_batch) & (trained_env.timestamp < harvest_time)].sort_values("timestamp").tail(SEQ_LEN).copy()
        root_row = root_rows.iloc[0] if len(root_rows) else {}
        varieties = ["TME 419", "TMS 30572", "Local sweet", "Local bitter", "Other/unknown"]
        conditions = ["Good", "Fair", "Poor", "Unknown"]
        default_var = str(root_row.get("variety", "Other/unknown"))
        default_initial = str(root_row.get("initial_condition", "Good"))
        default_damage = int(np.clip(root_row.get("mechanical_damage", 0), 0, 3))
        default_dry = float(np.clip(root_row.get("dry_matter_pct", 34), 0, 100))
        default_harvest_hour = int(harvest_time.hour) if pd.notna(harvest_time) else 0
        with st.form("existing_batch_prediction_form"):
            variety = st.selectbox("Cassava variety", varieties, index=varieties.index(default_var) if default_var in varieties else len(varieties)-1)
            damage = st.slider("Mechanical damage indicator (0 = none, 3 = severe)", 0, 3, default_damage)
            initial = st.selectbox("Initial root condition", conditions, index=conditions.index(default_initial) if default_initial in conditions else 0)
            dry = st.number_input("Dry matter content (%)", 0.0, 100.0, default_dry, 0.5)
            harvest_hour = st.number_input("Harvest time (hour of day, 0–23)", 0, 23, default_harvest_hour)
            submitted = st.form_submit_button("Estimate 72-hour deterioration risk")
        if submitted:
            try:
                forecast, risk = predict_batch(bundle, recent, variety, damage, initial, dry, harvest_hour)
                st.session_state.last_prediction = (selected_batch, forecast, risk)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    else:
        st.info("Manual entry is for predicting a new batch after the hybrid model has been trained using uploaded CSV data or the supplied demo dataset.")
        st.warning("The current Bi-LSTM was trained with temperature and relative humidity. Oxygen is collected below for the batch record, but it is not yet used by this model because the supplied training dataset and trained architecture do not contain oxygen measurements.")
        with st.form("manual_batch_prediction_form"):
            manual_batch_id = st.text_input("Batch ID", value="MANUAL-001")
            c1, c2 = st.columns(2)
            with c1:
                manual_temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=28.0, step=0.1)
                manual_rh = st.number_input("Relative humidity (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
                manual_oxygen = st.number_input("Oxygen level (%)", min_value=0.0, max_value=100.0, value=20.9, step=0.1)
            with c2:
                manual_variety = st.selectbox("Cassava variety", ["TME 419", "TMS 30572", "Local sweet", "Local bitter", "Other/unknown"])
                manual_damage = st.slider("Mechanical damage (0 = none, 3 = severe)", 0, 3, 1)
                manual_condition = st.selectbox("Initial root condition", ["Good", "Fair", "Poor", "Unknown"])
                manual_dry = st.number_input("Dry matter content (%)", min_value=0.0, max_value=100.0, value=34.0, step=0.5)
            manual_harvest_date = st.date_input("Harvest date")
            manual_harvest_time = st.time_input("Harvest time")
            manual_submit = st.form_submit_button("Predict manually entered batch")
        if manual_submit:
            try:
                if not manual_batch_id.strip():
                    raise ValueError("Please enter a batch ID.")
                manual_harvest = pd.Timestamp.combine(manual_harvest_date, manual_harvest_time)
                # The trained model expects 24 hourly temperature/RH observations.
                # A single manual reading is repeated to form a clearly disclosed constant-history input.
                manual_history = pd.DataFrame({
                    "timestamp": [manual_harvest - pd.Timedelta(hours=h) for h in range(SEQ_LEN, 0, -1)],
                    "temperature_c": [manual_temp] * SEQ_LEN,
                    "relative_humidity_pct": [manual_rh] * SEQ_LEN,
                })
                forecast, risk = predict_batch(
                    bundle, manual_history, manual_variety, manual_damage,
                    manual_condition, manual_dry, manual_harvest.hour,
                )
                st.session_state.last_prediction = (manual_batch_id.strip(), forecast, risk)
                st.session_state.last_manual_oxygen = manual_oxygen
                st.caption("Manual-input assumption: the entered temperature and humidity are treated as constant across the preceding 24 hours because a full hourly history was not supplied.")
            except Exception as exc:
                st.error(f"Manual prediction failed: {exc}")

    if st.session_state.get("last_prediction") is not None:
        result_batch, result_forecast, result_risk = st.session_state.last_prediction
        st.subheader(f"Prediction result: {result_batch}")
        st.metric("Deterioration Risk Index (DRI)", f"{result_risk * 100:.1f}/100")
        st.caption("DRI is the XGBoost model's estimated probability for the supplied binary 72-hour outcome, expressed as a percentage. It is not a universal biological scale.")
        if result_risk >= 0.70:
            st.error("Higher model-estimated risk. Prioritise inspection and consider prompt processing based on local procedures.")
        elif result_risk >= 0.40:
            st.warning("Moderate model-estimated risk. Review the batch condition and storage history.")
        else:
            st.success("Lower model-estimated risk relative to this trained model.")
        result_df = pd.DataFrame(result_forecast, columns=ENV_COLS)
        result_df.insert(0, "forecast_hour", np.arange(1, HORIZON + 1))
        st.subheader("Next 72-hour environmental forecast")
        st.line_chart(result_df.set_index("forecast_hour"))
        st.download_button("Download forecast CSV", result_df.to_csv(index=False), "cassava_72_hour_forecast.csv", "text/csv", key="download_latest_forecast")

else:
    st.info("Train the hybrid framework to enable the batch prediction section.")

with st.expander("Required CSV format"):
    st.markdown("**Environmental time-series CSV** (hourly records, one row per batch and timestamp)")
    st.code("batch_id,timestamp,temperature_c,relative_humidity_pct\nBATCH-001,2025-01-01 00:00:00,28.4,72.0\nBATCH-001,2025-01-01 01:00:00,28.1,73.2")
    st.markdown("**Cassava phenotype and outcome CSV** (one row per batch)")
    st.code("batch_id,variety,mechanical_damage,initial_condition,dry_matter_pct,ppd_status_72h,harvest_timestamp\nBATCH-001,TME 419,1,Good,34.5,0,2025-01-02 00:00:00")
    st.caption(
        "Each labelled batch needs 24 consecutive hourly environmental observations immediately before harvest, "
        "followed by 72 consecutive hourly observations after harvest. The binary outcome must be "
        "independently observed at the end of that post-harvest 72-hour period. Batch IDs must link the two files."
    )

st.divider()
st.caption(
    "Research boundary: this prototype predicts fresh cassava-root physiological post-harvest deterioration (PPD). "
    "It does not predict fufu spoilage, crop yield, or deterioration from images, and it does not connect to physical IoT hardware."
)
