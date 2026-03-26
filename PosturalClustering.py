# Input: animals must be a list of dictionaries, one per animal.
# Each dictionary must contain:
#   - "group": condition label (e.g., "Torpor" or "Sleep")
#   - "name": animal identifier
#   - "coord_path": path to an Excel file containing per-frame body-part coordinates
#
# Each Excel file must contain the following columns:
# x_nose, y_nose, x_R_ear, y_R_ear, x_L_ear, y_L_ear,
# x_back, y_back, x_R_hip, y_R_hip, x_L_hip, y_L_hip,
# x_tailbase, y_tailbase


import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


FPS = 12
BIN_SECONDS = 10
FRAMES_PER_BIN = FPS * BIN_SECONDS
IMMOBILE_THRESHOLD = 50
MIN_IMMOBILE_BINS = 4
N_GMM_COMPONENTS = 4

COORD_COLUMNS = [
    "x_nose", "y_nose",
    "x_R_ear", "y_R_ear",
    "x_L_ear", "y_L_ear",
    "x_back", "y_back",
    "x_R_hip", "y_R_hip",
    "x_L_hip", "y_L_hip",
    "x_tailbase", "y_tailbase",
]

REMOVE_FEATURES = ["nose_x", "move_back"]


def is_number(x):
    return isinstance(x, (int, float, np.number)) and pd.notna(x)


def get_point(row, xcol, ycol):
    x, y = row[xcol], row[ycol]
    return (float(x), float(y)) if is_number(x) and is_number(y) else None


def rotate_to_pos_y(vx, vy):
    return math.atan2(vx, vy)


def apply_rotation(x, y, phi):
    c, s = math.cos(phi), math.sin(phi)
    return c * x - s * y, s * x + c * y


def euclid_dist(p_now, p_prev):
    if p_now is None:
        return np.nan
    if p_prev is None:
        return 0.0
    dx, dy = p_now[0] - p_prev[0], p_now[1] - p_prev[1]
    return math.hypot(dx, dy)


def build_frame_features(df_coords):
    rows = []
    prev_back = None
    prev_nose = None

    for _, row in df_coords.iterrows():
        nose = get_point(row, "x_nose", "y_nose")
        back = get_point(row, "x_back", "y_back")
        r_ear = get_point(row, "x_R_ear", "y_R_ear")
        l_ear = get_point(row, "x_L_ear", "y_L_ear")
        r_hip = get_point(row, "x_R_hip", "y_R_hip")
        l_hip = get_point(row, "x_L_hip", "y_L_hip")
        tail = get_point(row, "x_tailbase", "y_tailbase")

        if back is not None:
            bx, by = back

            def shift(p):
                return None if p is None else (p[0] - bx, p[1] - by)

            nose_s, r_ear_s, l_ear_s, r_hip_s, l_hip_s, tail_s = map(
                shift, [nose, r_ear, l_ear, r_hip, l_hip, tail]
            )

            if nose_s is not None and (nose_s[0] != 0 or nose_s[1] != 0):
                phi = rotate_to_pos_y(*nose_s)
            else:
                phi = 0.0

            def rot(p):
                return (np.nan, np.nan) if p is None else apply_rotation(p[0], p[1], phi)

            nose_r, r_ear_r, l_ear_r, r_hip_r, l_hip_r, tail_r = map(
                rot, [nose_s, r_ear_s, l_ear_s, r_hip_s, l_hip_s, tail_s]
            )
        else:
            nose_r = r_ear_r = l_ear_r = r_hip_r = l_hip_r = tail_r = (np.nan, np.nan)

        rows.append(
            {
                "nose_x": nose_r[0],
                "nose_y": nose_r[1],
                "r_ear_x": r_ear_r[0],
                "r_ear_y": r_ear_r[1],
                "l_ear_x": l_ear_r[0],
                "l_ear_y": l_ear_r[1],
                "r_hip_x": r_hip_r[0],
                "r_hip_y": r_hip_r[1],
                "l_hip_x": l_hip_r[0],
                "l_hip_y": l_hip_r[1],
                "tailbase_x": tail_r[0],
                "tailbase_y": tail_r[1],
                "move_back": euclid_dist(back, prev_back),
                "move_nose": euclid_dist(nose, prev_nose),
            }
        )

        prev_back = back
        prev_nose = nose

    return pd.DataFrame(rows)


def make_bins_time_major(X, frames_per_bin):
    n_bins = X.shape[0] // frames_per_bin
    if n_bins == 0:
        return np.empty((0, X.shape[1]))
    X = X[: n_bins * frames_per_bin]
    return np.nanmean(X.reshape(n_bins, frames_per_bin, -1), axis=1)


def detect_immobile_bins(df_feat, frames_per_bin, threshold):
    move_back = df_feat["move_back"].to_numpy(dtype=float)
    n_bins = len(move_back) // frames_per_bin
    if n_bins == 0:
        return np.array([], dtype=bool)

    move_back = move_back[: n_bins * frames_per_bin]
    move_back_bin = np.nansum(move_back.reshape(n_bins, frames_per_bin), axis=1)
    return move_back_bin < threshold


def find_bouts(mask, min_len):
    bouts = []
    start = None

    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                bouts.append((start, i))
            start = None

    if start is not None and len(mask) - start >= min_len:
        bouts.append((start, len(mask)))

    return bouts


def build_animal_data(animals):
    animal_data = {}

    for a in animals:
        name = a["name"]
        coord_df = pd.read_excel(a["coord_path"], usecols=COORD_COLUMNS)
        feat_full = build_frame_features(coord_df)
        imm_mask = detect_immobile_bins(feat_full, FRAMES_PER_BIN, IMMOBILE_THRESHOLD)

        animal_data[name] = {
            "group": a["group"],
            "coord_df": coord_df,
            "feat_full": feat_full,
            "imm_mask": imm_mask,
        }

    return animal_data


def extract_bout_mean_features(animal_data):
    feat14_all = []
    bout_labels = []
    bout_names = []
    start_bins_all = []
    end_bins_all = []

    for name, data in animal_data.items():
        feat = data["feat_full"]
        bins = make_bins_time_major(feat.to_numpy(dtype=float), FRAMES_PER_BIN)
        imm = data["imm_mask"][:len(bins)]
        bouts = find_bouts(imm, MIN_IMMOBILE_BINS)

        for s, e in bouts:
            bout_mean = np.nanmean(bins[s:e], axis=0)
            feat14_all.append(bout_mean)
            bout_labels.append(data["group"])
            bout_names.append(name)
            start_bins_all.append(s)
            end_bins_all.append(e)

    if len(feat14_all) == 0:
        raise ValueError("No immobile bouts detected.")

    feat14_all = np.vstack(feat14_all)

    first_name = next(iter(animal_data))
    feature_names_14 = list(animal_data[first_name]["feat_full"].columns)

    keep_idx = [i for i, name in enumerate(feature_names_14) if name not in REMOVE_FEATURES]
    feature_names_12 = [feature_names_14[i] for i in keep_idx]
    feat12_all = feat14_all[:, keep_idx]

    return {
        "features_14": feat14_all,
        "features_12": feat12_all,
        "feature_names_14": feature_names_14,
        "feature_names_12": feature_names_12,
        "bout_labels": np.array(bout_labels),
        "bout_names": np.array(bout_names),
        "start_bins": np.array(start_bins_all),
        "end_bins": np.array(end_bins_all),
    }


def run_postural_clustering(animals):
    animal_data = build_animal_data(animals)
    bout_data = extract_bout_mean_features(animal_data)
    feat12_all = bout_data["features_12"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat12_all)

    pca = PCA(n_components=2)
    pca_scores = pca.fit_transform(X_scaled)

    gmm = GaussianMixture(
        n_components=N_GMM_COMPONENTS,
        covariance_type="full",
        n_init=30,
        random_state=0,
    )
    cluster_labels = gmm.fit_predict(pca_scores)

    return {
        "animal_data": animal_data,
        **bout_data,
        "pca_scores": pca_scores,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cluster_labels": cluster_labels,
        "pca": pca,
        "gmm": gmm,
        "scaler": scaler,
    }
