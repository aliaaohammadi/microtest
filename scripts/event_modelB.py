from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
)

# =============== USER SETTINGS ===============
data_folder = Path(r"C:\Users\amoh\Desktop\microtox\eventhallen\signals")

imu_columns = [
    "IMU-1 Axl.X", "IMU-1 Axl.Y", "IMU-1 Axl.Z",
    "IMU-1 Gyr.X", "IMU-1 Gyr.Y", "IMU-1 Gyr.Z",
]

fs = 10                 # Hz
window_sec = 5
overlap = 0.5           # IMPORTANT: no overlap
window_size = int(window_sec * fs)
step_size = int(window_size * (1 - overlap))  # equals window_size

# focus vehicle: VOI scooter
VOI_VEHICLE_CODE = 1

# sequential / detector settings
PRIOR_DRUNK = 0.10
THRESHOLD = 0.90
P_STAY = 0.995          # HMM stickiness

# ============================================


# =============== WINDOWING + FEATURES ===============
def iter_windows(df: pd.DataFrame, window_size: int, step_size: int):
    n = len(df)
    w_id = 0
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        yield w_id, df.iloc[start:end]
        w_id += 1


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def _band_energy(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    # simple FFT band energy (works fine for short windows)
    x = x - np.mean(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    psd = (np.abs(X) ** 2)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(psd[mask].sum())


def imu_features_for_window(wdf: pd.DataFrame, fs: float) -> dict:
    feats = {}

    # per-channel stats
    for c in imu_columns:
        x = wdf[c].to_numpy(dtype=float)
        feats[f"{c}_mean"] = float(np.mean(x))
        feats[f"{c}_std"]  = float(np.std(x))
        feats[f"{c}_rms"]  = _rms(x)
        feats[f"{c}_p2p"]  = float(np.ptp(x))
        feats[f"{c}_hfE"]  = _band_energy(x, fs, 2.0, min(4.0, fs/2 - 0.1))  # 2–4 Hz-ish

    # magnitudes
    ax = wdf["IMU-1 Axl.X"].to_numpy(float)
    ay = wdf["IMU-1 Axl.Y"].to_numpy(float)
    az = wdf["IMU-1 Axl.Z"].to_numpy(float)
    gx = wdf["IMU-1 Gyr.X"].to_numpy(float)
    gy = wdf["IMU-1 Gyr.Y"].to_numpy(float)
    gz = wdf["IMU-1 Gyr.Z"].to_numpy(float)

    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    feats["acc_mag_mean"] = float(np.mean(a_mag))
    feats["acc_mag_std"]  = float(np.std(a_mag))
    feats["acc_mag_rms"]  = _rms(a_mag)
    feats["gyr_mag_mean"] = float(np.mean(g_mag))
    feats["gyr_mag_std"]  = float(np.std(g_mag))
    feats["gyr_mag_rms"]  = _rms(g_mag)

    return feats


# =============== HMM FORWARD FILTER (BINARY) ===============
def make_binary_transition(p_stay: float) -> np.ndarray:
    p_sw = 1.0 - p_stay
    return np.array([[p_stay, p_sw],
                     [p_sw,   p_stay]], dtype=float)

def hmm_forward_binary(emission_lik: np.ndarray, A: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """
    emission_lik: (T,2) proportional to p(x_t|y)
    A: (2,2) transition with rows sum to 1, A[i,j]=P(y_t=j|y_{t-1}=i)
    prior: (2,)
    """
    T = emission_lik.shape[0]
    post = np.zeros((T, 2), float)

    alpha = prior * emission_lik[0]
    alpha = alpha / (alpha.sum() + 1e-12)
    post[0] = alpha

    for t in range(1, T):
        pred = post[t-1] @ A
        alpha = pred * emission_lik[t]
        post[t] = alpha / (alpha.sum() + 1e-12)

    return post


# =============== 1) BUILD WINDOW DATASET ===============
def build_window_df(data_folder: Path, normalize_per_file: bool = True) -> pd.DataFrame:
    records = []

    for file in data_folder.glob("*.csv"):
        name = file.stem
        parts = name.split("_")
        if len(parts) < 5:
            continue

        participant = int(parts[0][1:])    # Pxxx
        intox_level = int(parts[1][1:])    # Lx
        vehicle     = int(parts[2][1:])    # Vx
        repetition  = int(parts[3][1:])    # Rx
        task        = parts[4]             # task string

        if vehicle != VOI_VEHICLE_CODE:
            continue

        df = pd.read_csv(file)

        # column check
        missing = [c for c in imu_columns if c not in df.columns]
        if missing:
            print(f"Missing columns in {file.name}: {missing}")
            continue

        # normalize within file (optional)
        if normalize_per_file:
            df[imu_columns] = (df[imu_columns] - df[imu_columns].mean()) / (df[imu_columns].std() + 1e-6)

        # windows
        for w_id, wdf in iter_windows(df, window_size, step_size):
            feats = imu_features_for_window(wdf, fs)

            rec = {
                "participant": participant,
                "intox_level": intox_level,
                "vehicle": vehicle,
                "repetition": repetition,
                "task": task,
                "window": w_id,
                "file": file.name,
                "y_true": int(intox_level > 0),   # binary
            }
            rec.update(feats)
            records.append(rec)

    return pd.DataFrame(records)


# =============== 2) MODEL (XGBoost if available, else fallback) ===============
def make_window_model():
    try:
        from xgboost import XGBClassifier
        base = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        # XGB doesn't need scaling, but calibration helps a lot for sequential Bayes
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        return model
    except Exception as e:
        # fallback: scaled logistic regression + calibration
        from sklearn.linear_model import LogisticRegression
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))
        ])
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        return model


# =============== 3) LOPO EVAL + SEQUENTIAL ACCUMULATION ===============
def evaluate_lopo_with_hmm(feat_df: pd.DataFrame):
    meta_cols = {"participant","intox_level","vehicle","repetition","task","window","file","y_true"}
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]

    logo = LeaveOneGroupOut()
    groups = feat_df["participant"].to_numpy()
    y = feat_df["y_true"].to_numpy()

    A = make_binary_transition(P_STAY)
    init_prior = np.array([1.0 - PRIOR_DRUNK, PRIOR_DRUNK], float)

    all_win_true, all_win_pred = [], []
    all_trial_true, all_trial_pred = [], []
    all_alarm_true, all_alarm_pred = [], []

    for fold, (tr_idx, te_idx) in enumerate(logo.split(feat_df, y, groups=groups), start=1):
        train = feat_df.iloc[tr_idx].copy()
        test  = feat_df.iloc[te_idx].copy()

        # training prior for likelihood conversion
        y_tr = train["y_true"].to_numpy()
        pi = np.array([np.mean(y_tr == 0), np.mean(y_tr == 1)], float)
        pi = pi / (pi.sum() + 1e-12)

        model = make_window_model()
        model.fit(train[feature_cols].to_numpy(), y_tr)

        # run sequential per (file) sequence
        seq_out = []
        for file_name, df_seq in test.groupby("file"):
            df_seq = df_seq.sort_values("window")
            X = df_seq[feature_cols].to_numpy()

            q = model.predict_proba(X)  # q(y|x)
            # convert to emission likelihood ~ q/pi
            lik = np.clip(q / (pi.reshape(1,2) + 1e-12), 1e-12, None)

            post = hmm_forward_binary(lik, A, init_prior)

            tmp = df_seq.copy()
            tmp["q_drunk"] = q[:, 1]
            tmp["post_drunk"] = post[:, 1]
            seq_out.append(tmp)

        seq_test = pd.concat(seq_out, ignore_index=True)

        # window-level
        y_true_w = seq_test["y_true"].to_numpy()
        y_pred_w = (seq_test["post_drunk"].to_numpy() >= 0.5).astype(int)
        all_win_true.append(y_true_w)
        all_win_pred.append(y_pred_w)

        # trial-level: last window of each file
        last = (seq_test.sort_values(["file","window"])
                        .groupby("file", as_index=False)
                        .tail(1))
        y_true_t = last["y_true"].to_numpy()
        y_pred_t = (last["post_drunk"].to_numpy() >= 0.5).astype(int)
        all_trial_true.append(y_true_t)
        all_trial_pred.append(y_pred_t)

        # alarm-level: any time >= THRESHOLD
        mx = seq_test.groupby("file")["post_drunk"].max()
        y_alarm_pred = (mx.to_numpy() >= THRESHOLD).astype(int)
        y_alarm_true = seq_test.groupby("file")["y_true"].first().to_numpy().astype(int)
        all_alarm_true.append(y_alarm_true)
        all_alarm_pred.append(y_alarm_pred)

        print(f"[Fold {fold:02d}] held-out P{int(test['participant'].iloc[0]):03d} | "
              f"trial bal-acc: {balanced_accuracy_score(y_true_t, y_pred_t):.3f} | "
              f"alarm bal-acc: {balanced_accuracy_score(y_alarm_true, y_alarm_pred):.3f}")

    # aggregate + report
    yW = np.concatenate(all_win_true); pW = np.concatenate(all_win_pred)
    yT = np.concatenate(all_trial_true); pT = np.concatenate(all_trial_pred)
    yA = np.concatenate(all_alarm_true); pA = np.concatenate(all_alarm_pred)

    print("\n=== WINDOW (posterior>=0.5) ===")
    print("Accuracy:", accuracy_score(yW, pW))
    print("Balanced accuracy:", balanced_accuracy_score(yW, pW))
    print(classification_report(yW, pW, target_names=["Sober","Drunk"]))

    print("\n=== TRIAL (last posterior>=0.5) ===")
    print("Accuracy:", accuracy_score(yT, pT))
    print("Balanced accuracy:", balanced_accuracy_score(yT, pT))
    print(classification_report(yT, pT, target_names=["Sober","Drunk"]))
    print("Confusion:\n", confusion_matrix(yT, pT))

    print(f"\n=== ALARM (any posterior>={THRESHOLD:.2f}) ===")
    print("Accuracy:", accuracy_score(yA, pA))
    print("Balanced accuracy:", balanced_accuracy_score(yA, pA))
    print(classification_report(yA, pA, target_names=["Sober","Drunk"]))
    print("Confusion:\n", confusion_matrix(yA, pA))


# =============== RUN ===============
if __name__ == "__main__":
    feat_df = build_window_df(data_folder, normalize_per_file=True)
    print("Window dataset shape:", feat_df.shape)
    print("Class balance:\n", feat_df["y_true"].value_counts())

    evaluate_lopo_with_hmm(feat_df)