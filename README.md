
# ML Fitness Tracking – Barbell Exercise Classification 🏋️‍♂️🤖

This is an end-to-end machine learning project focused on classifying barbell exercises using wearable sensor data.  blending real-world AI applications with something I am passionate about: **fitness**.

## 🚀 Why this project?

I embarked on this project as part of my development journey to explore and research what we can achieve with Machine Learning (ML) and AI in the fitness domain. I discovered this great project by Dave Ebbelaar that walks through the structure and logic of building an ML pipeline — **but doesn’t provide any starter code**. 

All core development you see here has been written from scratch by me (with only a few functions borrowed from existing libraries to help streamline certain parts of the process). I took the time to dive deep into key concepts and techniques for feature engineering, model building, and adding useful functions, all while maintaining a mindset focused on understanding not just the WHAT or HOW, but also the WHY behind every step.

> 🧠 **Learning-by-doing** is the best way to internalize concepts and truly understand the material.

This approach has helped me:
- **Deepen my understanding of ML workflows** through real-world application.
    - Speacially on time-series sensor data for machine learning.
- **Sharpen my coding and debugging skills** by solving problems as they arise
- **Stay motivated and engaged** by working on a project that combines something I'm passionate about: fitness and machine learning

> ⚠️ No shortcuts. No pre-written code. Just pure hands-on learning and problem-solving.

## 📌 Why this matters to me

I go to the gym often and this project hits that sweet spot between **technical growth** and **personal interest**. It helped me level up for professional skills and knowledge, gaining practical experience that I can apply to similar challenges in my career and future projects, all while staying deeply engaged.

> This repo isn’t about perfection — it’s about progression.
> Every commit is part of the learning journey.

## 📝 Credits

This project follows the structure and logic from Dave Ebbelaar's Full Machine Learning Project guide on building an ML pipeline. While no starter code was provided, his detailed explanations and videos served as the blueprint for the work. 

---

## 🧠 What’s this project about?

The goal is to build a system that can classify different **barbell exercises** using time-series data from motion sensors (IMU). This includes:

- Signal preprocessing
- Feature engineering
- Model training & evaluation
- Real-time prediction pipeline

The project combines data science, machine learning, and signal processing in a way that mirrors industry-level challenges.

## 🔍 Tech Stack

- Python (NumPy, Pandas, Scikit-learn, etc.)
- ML models: TBD (depending on the current stage)
- Data from wearable sensors (accelerometers, gyroscopes)
- VS Code and Jupyter Notebooks for development and experimentation



## Data Science Project Template

You can use this template to structure your Python data science projects. It is based on [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).


---

# Project Details


## Build Features:

## Butterworth lowpass filter

Smooth raw movement data (e.g., accelerometer) to highlight the core motion patterns (e.g., reps), while removing subtle noise and jitter.
The filter doesn’t isolate reps — it works on the full time-series signal.
It removes anything that changes **too quickly** (high frequency), and keeps **slow changes** (real rep motion). 

### 🎯 Why Use a Lowpass Filter?
- Movement patterns (like reps) are low frequency. Sensor noise is usually high frequency.
- The filter keeps the slow, meaningful patterns and removes fast, noisy fluctuations

## Dimensionality Reduction with PCA

After noise reduction, we apply **Principal Component Analysis (PCA)** to:

- Reduce the number of feature predictors
- Preserve the majority of the variance (i.e., key signal patterns)
- Simplify the dataset while retaining its essential structure

This is especially useful for downstream modeling and visualization of movement trends across reps and participants.

## Sum of Squares Features (Movement Intensity)

To capture overall movement energy—independent of device orientation, we have to combine the three axes of both accelerometer and gyroscope into single scalar magnitudes. 

- acc_r: overall acceleration magnitude (acc_x² + acc_y² + acc_z²)
- gyr_r: overall rotational magnitude (gyr_x² + gyr_y² + gyr_z²)

These are **direction-agnostic** features — useful when:
- Participants move in slightly different directions
- A more universal signal of movement intensity/ Sensor orientation isn’t perfectly consistent
- Better performance in real-world scenarios. You want features that generalize better across users

> Value: "Equalizer" across users. It helps reduce orientation bias and makes your features more robust for ML modeling.

## ⏱ Temporal Abstraction

To prepare time-series sensor data for machine learning, we apply **temporal abstraction**, which summarizes movement over short time windows using statistical features like:

- **Mean** – the average signal value over the window
- **Standard deviation** – the signal’s variability over the window

### 🎯 Why Temporal Abstraction?

Raw sensor data (e.g., from accelerometers and gyroscopes) is highly detailed and noisy, collected at millisecond intervals. Temporal abstraction:

- Reduces noise and smooths fluctuations
- Converts chaotic signal spikes into structured, stable features
- Preserves the **core motion patterns** like reps or transitions
- Makes the data more useful for ML models

### ⚙️ How It Works?

We use a **sliding window** approach:

- The window size is ~1 second (5 samples, given 200ms sampling rate)
- For each row, we compute the mean and std over the surrounding window
- The dataset keeps the **same number of rows** (i.e., per-millisecond resolution), just with new abstracted columns

This results in features like:

| Time | acc_y | acc_y_mean | acc_y_std | acc_r_mean | acc_r_std |
|------|-------|------------|-----------|------------|------------|
| ...  | ...   | ...        | ...       | ...        | ...        |

> ✅ Each row now carries context about the recent movement — perfect for models that require temporal continuity.

### Why Include `acc_r` and `gyr_r`?

We also compute abstraction on:
- `acc_r`: combined intensity of all accelerometer axes
- `gyr_r`: combined intensity of all gyroscope axes

By adding `mean` and `std` of these composite values, we capture **how strong and consistent** the overall movement is 

In short:
> **Temporal abstraction turns noisy raw signals into powerful movement descriptors** – helping your model focus on *what matters*.
ns.

## Frequency Features (Fourier Transform)

To capture the *repetition patterns* in movement (e.g., squats, push-ups), we apply a **Fourier Transform** to the time-series data.

**Why?**  
Repetitive motions have distinct frequencies (e.g., a rep every 2.5 sec = ~0.4 Hz).  
Noise and jitter appear at higher frequencies.

**What it does:**  
Transforms movement data from time → frequency domain, giving us:

- **Dominant repetition frequency**  
- **Power and consistency** of movement rhythm  
- Filters out subtle noise or irregularities

These features help the model better understand movement quality and tempo — which can vary across exercises and participants.


## Dealing with Overlapping Windows

When using sliding windows to compute temporal and frequency-based features, overlapping windows can create highly correlated samples. This artificial similarity between rows can lead to **model overfitting** — where the model learns repetitive patterns rather than meaningful variation.

✅ **Solution**: 50% Downsample is considere a good threshold if there's enough data. Downsampling the data by keeping only every second row after feature extraction. 

This reduces the correlation between consecutive samples, leading to:
- Better model generalization
- Less overfitting during training
- More realistic performance on new, unseen data
