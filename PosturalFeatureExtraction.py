import pandas as pd
import numpy as np
from math import degrees


# Input: Excel file with per-frame body-part coordinates.
# Expected columns: x_nose, y_nose, x_R_ear, y_R_ear, ..., x_tailbase, y_tailbase
FILE_PATH = "/Users/CP.xlsx"
YMAX = 720


def calculate_distance(p1, p2, ymax=YMAX):
    dx = p1[0] - p2[0]
    dy = (ymax - p1[1]) - (ymax - p2[1])
    return np.sqrt(dx**2 + dy**2)


def calculate_angle(p1, p2, p3, ymax=YMAX):
    v1 = np.array(
        [p1[0] - p2[0], (ymax - p1[1]) - (ymax - p2[1])],
        dtype=float,
    )
    v2 = np.array(
        [p3[0] - p2[0], (ymax - p3[1]) - (ymax - p2[1])],
        dtype=float,
    )

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return np.nan

    unit_v1 = v1 / norm1
    unit_v2 = v2 / norm2
    dot_product = np.dot(unit_v1, unit_v2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))

    return degrees(angle_rad)


def calculate_speed(current, previous, ymax=YMAX):
    return calculate_distance(current, previous, ymax=ymax)


def extract_points(row):
    return {
        "nose": (row["x_nose"], row["y_nose"]),
        "right_ear": (row["x_R_ear"], row["y_R_ear"]),
        "left_ear": (row["x_L_ear"], row["y_L_ear"]),
        "back": (row["x_back"], row["y_back"]),
        "right_hip": (row["x_R_hip"], row["y_R_hip"]),
        "left_hip": (row["x_L_hip"], row["y_L_hip"]),
        "tailbase": (row["x_tailbase"], row["y_tailbase"]),
    }


def compute_postural_parameters(data, ymax=YMAX):
    parameters = []

    first_row = data.iloc[0]
    points = extract_points(first_row)
    previous_back = points["back"]

    parameters.append(
        {
            "Frame": 1,
            "Body Length": calculate_distance(points["nose"], points["back"], ymax=ymax)
            + calculate_distance(points["back"], points["tailbase"], ymax=ymax),
            "Body Width": calculate_distance(
                points["right_hip"], points["left_hip"], ymax=ymax
            ),
            "Head Angle": calculate_angle(
                points["right_ear"], points["nose"], points["left_ear"], ymax=ymax
            ),
            "Body Bend Angle": 180
            - calculate_angle(
                points["nose"], points["back"], points["tailbase"], ymax=ymax
            ),
            "Body Speed": 0,
        }
    )

    for i, row in data.iloc[1:].iterrows():
        points = extract_points(row)

        parameters.append(
            {
                "Frame": i + 1,
                "Body Length": calculate_distance(
                    points["nose"], points["back"], ymax=ymax
                )
                + calculate_distance(points["back"], points["tailbase"], ymax=ymax),
                "Body Width": calculate_distance(
                    points["right_hip"], points["left_hip"], ymax=ymax
                ),
                "Head Angle": calculate_angle(
                    points["right_ear"], points["nose"], points["left_ear"], ymax=ymax
                ),
                "Body Bend Angle": 180
                - calculate_angle(
                    points["nose"], points["back"], points["tailbase"], ymax=ymax
                ),
                "Body Speed": calculate_speed(
                    points["back"], previous_back, ymax=ymax
                ),
            }
        )

        previous_back = points["back"]

    return pd.DataFrame(parameters)


def main():
    data = pd.read_excel(FILE_PATH)
    parameters_df = compute_postural_parameters(data, ymax=YMAX)
    print(parameters_df.head())


if __name__ == "__main__":
    main()
