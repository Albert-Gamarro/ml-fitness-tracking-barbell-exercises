import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")

predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines .linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()  # non-null values for each predictor column now

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

"""
GOAL: Prepare for applying the Butterworth lowpass filter.

We're focusing on movement patterns during each exercise set.
To filter out subtle noise while keeping the meaningful repetition shapes,
we need to estimate the average repetition duration first.

This will help us set the correct cutoff frequency for the filter.
"""


# ? Why?
# ? The goal is to smooth the signal (sensor data) to remove subtle,
# ? high-frequency noise — while preserving the main movement pattern

df[df["set"] == 25]["acc_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start

    df.loc[(df["set"] == s), "duration"] = (
        duration.seconds
    )  # select the set, store duration in a new duration column

duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0]  # avg duration of heavy sets
duration_df.iloc[1]  # avg duration of medium sets
duration_df.iloc[0] / 5  # avg duration of heavy repetition
duration_df.iloc[1] / 10  # avg duration of medium repetition

"""
Understanding the repetition frequency:
    - If each rep takes about 2.5 to 3.0 seconds,
     then the *rep frequency* is:
        freq = 1 / rep_duration
             = 1 / 2.5 to 1 / 3
             ≈ 0.33 to 0.4 Hz
    - This means reps repeat about 0.33Hz – 0.4Hz times per second
"""


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

"""
We want to smooth the sensor signal using a lowpass filter
to remove small noisy fluctuations, while keeping the main repetition patterns.

- Set sampling frequency: 1000 / 200 = 5 Hz → we have one row every 200ms based on our data
- The cutoff defines the "speed limit" for which frequencies to keep
- The order → how "steep" the filter is (higher = more aggressive drop-off), by default is 5

"""

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000 / 200
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 25]

# ? comparing original column with smoothen value using LowPassFilter()
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(subset["acc_y"].reset_index(drop=True))
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True))


for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# ? Elbow Technique to determine the optimum number of PC
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("Principal component number")
plt.ylabel("explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)


subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

# acc_r: overall acceleration magnitude (acc_x² + acc_y² + acc_z²)
# gyr_r: overall rotational magnitude (gyr_x² + gyr_y² + gyr_z²)

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)


df_squared_subset = df_squared[df_squared["set"] == 25]
plt.figure(figsize=(20, 6))
plt.plot(df_squared_subset["acc_x"].reset_index(drop=True))
plt.plot(df_squared_subset["acc_r"].reset_index(drop=True))


df_squared_subset[["acc_r", "gyr_r"]].plot(subplots=True)
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
