# Data sources 
#   - parameters_df: parameter datasets obtained with PosturalFeatureExtraction.py
#                    For each parameter, values from all mice are concatenated in order

import numpy as np

# 【Step 1】: Randomly split training and test individuals
individuals = np.arange(11)
np.random.seed(42)
np.random.shuffle(individuals)
train_individuals = individuals[:6]
test_individuals = individuals[6:]
num_train_individuals = len(train_individuals)
num_test_individuals = len(test_individuals)
num_train_individuals = len(train_individuals)

# 【Step 2】: Extract training mice parameter data
individual_size = 1036800 
train_data_indices = []
for individual in train_individuals:
    start_idx = individual * individual_size
    end_idx = start_idx + individual_size
    train_data_indices.extend(range(start_idx, end_idx))
train_data_parameters = parameters_df.iloc[train_data_indices]

# 【Step 3】: Assign mobility state 
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

train_avg_parameters = train_data_parameters.groupby(np.arange(len(train_data_parameters)) // (12 * 60 * 5)).mean()
avg_individual_size = len(train_avg_parameters) // num_train_individuals   # = 288
train_norm_parameters = train_avg_parameters.copy()  
for i in range(num_train_individuals):
    start_param_idx = i * avg_individual_size
    end_param_idx = start_param_idx + avg_individual_size
    individual_parameters = train_avg_parameters.iloc[start_param_idx:end_param_idx]
    # Normalization
    scaler = MinMaxScaler()
    train_norm_parameters.iloc[start_param_idx:end_param_idx] = scaler.fit_transform(individual_parameters)

train_individual_parameters = pd.DataFrame()
for i in range(num_train_individuals):
    start_param_idx = i * avg_individual_size
    end_param_idx = start_param_idx + avg_individual_size
    individual_parameters = train_norm_parameters.iloc[start_param_idx:end_param_idx]
    # Kernel density estimation and threshold calculation
    data_kernel = individual_parameters['Body Speed'].dropna()
    # Performing kernel density estimation
    kde = gaussian_kde(data_kernel, bw_method='scott')
    x_range = np.linspace(data_kernel.min(), data_kernel.max(), 1000)
    kde_values = kde(x_range)
    smoothed_kde = gaussian_filter1d(kde_values, sigma=1)
    #Identify the threshold by finding the inflection point
    kde_diff = np.diff(smoothed_kde)
    peaks, _ = find_peaks(-kde_diff)
    threshold_index = peaks[0] if peaks.size > 0 else len(x_range) // 2
    threshold_value = x_range[threshold_index]
    # Categorizing data based on the threshold
    individual_parameters['category'] = individual_parameters['Body Speed'].apply(lambda x: 'immobile' if x < threshold_value else 'mobile')
    # Append individual data to the aggregated DataFrame
    train_individual_parameters = pd.concat([train_individual_parameters, individual_parameters], ignore_index=True)
    
# 【Step 4】: Downsample, flatten into bins, and perform PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def downsample_and_flatten_bins(data, original_freq=12, target_freq=3, seconds_per_bin=10):  # 1 bin = 10 sec
    step = int(original_freq / target_freq)
    downsampled_data = data.iloc[::step, :]
    bin_size = target_freq * seconds_per_bin
    num_bins = len(downsampled_data) // bin_size
    flattened = [downsampled_data.iloc[i * bin_size:(i + 1) * bin_size].values.flatten() for i in range(num_bins)]
    return pd.DataFrame(flattened)
    
train_data_pca = train_data_parameters[['Body Length', 'Body Width', 'Head Angle']]
# Flatten into bins
flattened_data = downsample_and_flatten_bins(train_data_pca)
# Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(flattened_data)
# Initialize and run PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(standardized_data)

# categoryを30行複製し、PCA成分とともにtrain_pca_resultに保存
categories = np.repeat(train_individual_parameters_fast['category'], 30)
train_pca_result = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
categories_series = pd.Series(categories[:len(train_pca_result)])
train_pca_result['category'] = categories_series.values


# 【Step 5】: Parameter tuning by leave-one-mouse-out cross-validation
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import (
    roc_curve, precision_score, recall_score, f1_score, accuracy_score
)
# 5-a) Data preparation
rows_per_mouse = len(train_pca_result) // num_train_individuals
train_pca = train_pca_result.copy()
train_pca['mouse_id'] = np.repeat(np.arange(num_train_individuals), rows_per_mouse)

X      = train_pca[['PC1', 'PC2']]
y      = train_pca['category'].map({'immobile': 1, 'mobile': 0})
groups = train_pca['mouse_id']

# 5-b) Perform nested LOMO-CV: inner loop for hyperparameter search, outer loop for threshold selection and metric calculation
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
thresholds   = []
best_C_list  = []
precisions, recalls, f1s, accuracies = [], [], [], []
outer        = LeaveOneGroupOut()

for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups), 1):
    # --- Outer split: define training and test indices for this fold ---
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]
    groups_tr  = groups.iloc[train_idx]

    # --- Inner loop: grid-search to optimize regularization parameter C ---
    inner = LeaveOneGroupOut()
    grid  = GridSearchCV(
        SVC(kernel='linear', probability=True),
        param_grid,
        cv=inner.split(X_tr, y_tr, groups_tr),
        scoring='roc_auc',
        n_jobs=-1,
        refit=True
    )
    grid.fit(X_tr, y_tr)
    best_C = grid.best_params_['C']
    best_C_list.append(best_C)

    # --- Retrain on outer-training set and determine decision threshold via Youden’s J statistic ---
    model = SVC(kernel='linear', C=best_C, probability=True, random_state=42)
    model.fit(X_tr, y_tr)
    proba_idx = list(model.classes_).index(1)
    y_tr_score = model.predict_proba(X_tr)[:, proba_idx]
    fpr, tpr, thr = roc_curve(y_tr, y_tr_score, pos_label=1)
    thr_opt = thr[np.argmax(tpr - fpr)]
    thresholds.append(thr_opt)

    # --- Evaluate on outer test set and compute performance metrics ---
    y_te_score = model.predict_proba(X_te)[:, proba_idx]
    y_pred     = (y_te_score >= thr_opt).astype(int)

    prec  = precision_score(y_te, y_pred, pos_label=1) * 100
    rec   = recall_score   (y_te, y_pred, pos_label=1) * 100
    f1sco = f1_score       (y_te, y_pred, pos_label=1) * 100
    acc   = accuracy_score(y_te, y_pred)             * 100

    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1sco)
    accuracies.append(acc)

# 【Step 5】 Train the final model on all mice
final_C = grid.best_params_['C']           
final_thr = np.mean(thresholds)
final_model = SVC(kernel='linear', C=final_C, probability=True, random_state=42)
final_model.fit(X, y)
