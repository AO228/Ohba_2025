# Data sources 
#   - parameters_df: parameter datasets obtained with PosturalFeatureExtraction.py

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Average body speed every 5 minutes
avg_speed = parameters_df['Body Speed'].groupby(np.arange(len(parameters_df)) // (12 * 60 * 5)).mean()

# Performing kernel density estimation
kde = gaussian_kde(avg_speed, bw_method='scott')
x_range = np.linspace(avg_speed.min(), avg_speed.max(), 1000)
kde_values = kde(x_range)
smoothed_kde = gaussian_filter1d(kde_values, sigma=1)

#Identify the threshold by finding the inflection point
kde_diff = np.diff(smoothed_kde) 
peaks, _ = find_peaks(-kde_diff) # Find peaks in the negative first derivative  
threshold_index = peaks[0] if peaks.size > 0 else len(x_range) // 2 # Choose first inflection point as threshold
threshold_value = x_range[threshold_index]
