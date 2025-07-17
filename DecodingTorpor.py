# Apply trained SVM to test data
# Data sources 
#   - parameters_df: parameter datasets obtained with PosturalFeatureExtraction.py
#                    For each parameter, values from all mice are concatenated in order
#   - test_individuals: NumPy array of test‐mouse individual IDs defined in SupportVectorClassifier.py
#   - scaler: StandardScaler instance already fitted on training data obtained with SupportVectorClassifier.py
#   - pca: PCA instance already fitted on training data obtained with SupportVectorClassifier.py
#   - svm_model: trained classifier obtained with SupportVectorClassifier.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA              
from sklearn.svm import SVC

# 【Step 1】: Extract test mice parameter data
individual_size = 1036800  
test_data_indices = []
for individual in test_individuals:
    start_idx = individual * individual_size
    end_idx = start_idx + individual_size
    test_data_indices.extend(range(start_idx, end_idx))

test_data_parameters = parameters_df.iloc[test_data_indices]
test_data_pca = test_data_parameters[['Body Length', 'Body Width', 'Head Angle']]

# 【Step 2】: Downsample, flatten into bins, and project test data into training PCA space
def downsample_and_flatten_bins(data, original_freq=12, target_freq=3, seconds_per_bin=10):  # 1 bin = 10 sec
    step = int(original_freq / target_freq)
    downsampled_data = data.iloc[::step, :]
    bin_size = target_freq * seconds_per_bin
    num_bins = len(downsampled_data) // bin_size
    flattened = [downsampled_data.iloc[i * bin_size:(i + 1) * bin_size].values.flatten() for i in range(num_bins)]
    return pd.DataFrame(flattened)

#　Apply downsampling and flattening to test mouse data
flattened_test_data = downsample_and_flatten_bins(test_data_pca)
# Normalize test data using the scaler fitted on training data (no .fit here!)
standardized_test_data = scaler.transform(flattened_test_data)
# Project standardized test data into the PCA space computed from training mice
test_principal_components = pca.transform(standardized_test_data)
test_pca_result = pd.DataFrame(test_principal_components, columns=['PC1', 'PC2'])

# 【Step 3】: Classify test data using the trained SVM model
# Extract PC1 and PC2 as input features for classification
X_test = test_pca_result[['PC1', 'PC2']]
# Predict mobility state for each bin using the trained SVM model
y_pred_svm = svm_model.predict(X_test)  
