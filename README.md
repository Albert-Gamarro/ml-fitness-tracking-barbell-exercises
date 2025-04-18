

# ML Fitness Tracking – Barbell Exercise Classification 🏋️‍♂️🤖

This is an end-to-end machine learning project focused on classifying barbell exercises using wearable sensor data.  blending real-world AI applications with something I am passionate about: **fitness**.

## 🚀 Why this project?

I embarked on this project as part of my development journey to explore and research what we can achieve with Machine Learning (ML) and AI in the fitness domain. I discovered this great project by Dave Ebbelaar that walks through the structure and logic of building an ML pipeline — **but doesn’t provide any starter code**. 

All core development you see here has been written from scratch by me (with only a few functions borrowed from existing libraries to help streamline certain parts of the process). I took the time to dive deep into key concepts and techniques for feature engineering, model building, and adding useful functions, all while maintaining a mindset focused on understanding not just the WHAT or HOW, but also the WHY behind every step.

> 🧠 **Learning-by-doing** is the best way to internalize concepts and truly understand the material.

This approach has helped me:
- **Deepen my understanding of ML workflows** through real-world application
- **Sharpen my coding and debugging skills** by solving problems as they arise
- **Stay motivated and engaged** by working on a project that combines something I'm passionate about: fitness and machine learning

> ⚠️ No shortcuts. No pre-written code. Just pure hands-on learning and problem-solving.

## 📌 Why this matters to me

I go to the gym often and this project hits that sweet spot between **technical growth** and **personal interest**. It helped me level up for professional skills and knowledge, gaining practical experience that I can apply to similar challenges in my career and future projects, all while staying deeply engaged.

> This repo isn’t about perfection — it’s about progression.
> Every commit is part of the learning journey.

## 📝 Credits

This project follows the structure and logic from Dave Ebbelaar's guide on building an ML pipeline. While no starter code was provided, his detailed explanations served as the blueprint for the work. All code and implementation were written by me, following his step-by-step approach
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



## Project Details


Build Features:

### Butterworth lowpass filter

Smooth raw movement data (e.g., accelerometer) to highlight the core motion patterns (e.g., reps), while removing subtle noise and jitter.
The filter doesn’t isolate reps — it works on the full time-series signal.
It removes anything that changes **too quickly** (high frequency), and keeps **slow changes** (real rep motion). 

🎯 Why Use a Lowpass Filter?
- Movement patterns (like reps) are low frequency. Sensor noise is usually high frequency.
- The filter keeps the slow, meaningful patterns and removes fast, noisy fluctuations

### 📉 Dimensionality Reduction with PCA

After noise reduction, we apply **Principal Component Analysis (PCA)** to:

- Reduce the number of feature predictors
- Preserve the majority of the variance (i.e., key signal patterns)
- Simplify the dataset while retaining its essential structure

This is especially useful for downstream modeling and visualization of movement trends across reps and participants.

## Sum of Squares Features (Movement Intensity)

To capture overall movement energy—independent of device orientation, we have to combine the three axes of both accelerometer and gyroscope into single scalar magnitudes. 

- acc_r: overall acceleration magnitude (acc_x² + acc_y² + acc_z²)
- gyr_r: overall rotational magnitude (gyr_x² + gyr_y² + gyr_z²)

Value: "Equalizer" across users. It helps reduce orientation bias and makes your features more robust for ML modeling.
- Direction-independent magnitude — you're not betting on one axis
- A more universal signal of movement intensity
- Better performance in real-world scenarios, especially with diverse participants or use cases