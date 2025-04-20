import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

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

""" 
Temporal Abstraction (mean & std per 1s window)
This smooths the raw time-series and summarizes motion per set
Result: structured features like acc_y_mean, gyr_z_std, acc_r_mean, etc.
"""

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()  # Initiating a new class instance

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = int(1000 / 200)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(
    df_temporal_list
)  # bringing everything back together into 1 dataframe

df_temporal[df_temporal["set"] == 25][
    ["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]
].plot()
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = (
    df_temporal.copy().reset_index()
)  # functions we are going ot use expect a discrete as an index
FreqAbs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2800 / 200)  # avg range of a repetition 2.8s

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
df_freq.columns
# Visualize results
subset_df_freq = df_freq[df_freq["set"] == 25]
subset_df_freq["acc_y"].plot()
subset_df_freq[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

# loop by set and variable

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformation to set {s} ")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(
        subset, predictor_columns, ws, fs
    )  # no need for col loop bc we can put a list (predictor columns)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2]  # every second row

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias_list = []


# Step 1: Try different K (number of clusters) and store the SSD (inertia)

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias_list.append(kmeans.inertia_)

# Step 2: Plot the Elbow Curve to find the point where increasing K
plt.figure(figsize=(20, 10))
plt.plot(k_values, inertias_list)
plt.xlabel("K")
plt.ylabel("Sum of Square distances (SSD)")
plt.show()


kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Visualize cluster
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# Plot accelerometer data to compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
