# Data sources
#   - CP.xlsx : Excel file containing the (x,y) coordinates of each body part
                # Each row corresponds to one video frame
                # Columns are organized as: x_nose, y_nose, x_R_ear, y_R_ear, ..., x_tailbase, y_tailbase

# 【Step 1】：Load body part coordinates from Excel
import pandas as pd
import numpy as np
from numpy.linalg import norm
from math import degrees, acos

file_path = '/Users/CP.xlsx'
data = pd.read_excel(file_path)
ymax = 720  # image height in pixels (for y-axis inversion)

# 【Step 2】：Define functions for postural parameter calculations
def calculate_distance(p1, p2): 
    """Euclidean distance with y-axis inversion"""
    return np.sqrt((p1[0] - p2[0])**2 + ((ymax - p1[1]) - (ymax - p2[1]))**2)

def calculate_angle(p1, p2, p3):
    """Angle at point p2 formed by vectors p1→p2 and p3→p2"""
    v1 = np.array([p1[0] - p2[0], (ymax - p1[1]) - (ymax - p2[1])])
    v2 = np.array([p3[0] - p2[0], (ymax - p3[1]) - (ymax - p2[1])])
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return degrees(angle_rad)

def calculate_speed(current, previous):
    """Frame-to-frame displacement of the back point"""
    return calculate_distance(current, previous)

# 【Step 3】：Compute postural parameters for each frame
parameters = []
# First frame initialization
first = data.iloc[0]
points = {
    1: (first['x_nose'], first['y_nose']),
    2: (first['x_R_ear'], first['y_R_ear']),
    3: (first['x_L_ear'], first['y_L_ear']),
    4: (first['x_back'], first['y_back']),
    5: (first['x_R_hip'], first['y_R_hip']),
    6: (first['x_L_hip'], first['y_L_hip']),
    7: (first['x_tailbase'], first['y_tailbase'])
}
previous_back = points[4]

parameters.append({
    'Frame': 1,
    'Body Length': calculate_distance(points[1], points[4]) + calculate_distance(points[4], points[7]),
    'Body Width': calculate_distance(points[5], points[6]),
    'Head Angle': calculate_angle(points[2], points[1], points[3]),
    'Body Bend Angle': 180 - calculate_angle(points[1], points[4], points[7]),
    'Body Speed': 0
})

# Remaining frames
for i, row in data.iloc[1:].iterrows():
    points = {
        1: (row['x_nose'], row['y_nose']),
        2: (row['x_R_ear'], row['y_R_ear']),
        3: (row['x_L_ear'], row['y_L_ear']),
        4: (row['x_back'], row['y_back']),
        5: (row['x_R_hip'], row['y_R_hip']),
        6: (row['x_L_hip'], row['y_L_hip']),
        7: (row['x_tailbase'], row['y_tailbase'])
    }
    parameters.append({
        'Frame': i + 1,
        'Body Length': calculate_distance(points[1], points[4]) + calculate_distance(points[4], points[7]),
        'Body Width': calculate_distance(points[5], points[6]),
        'Head Angle': calculate_angle(points[2], points[1], points[3]),
        'Body Bend Angle': 180 - calculate_angle(points[1], points[4], points[7]),
        'Body Speed': calculate_speed(points[4], previous_back)
    })

    previous_back = points[4]

# 【Step 4】：Convert parameter list to DataFrame
parameters_df = pd.DataFrame(parameters)

