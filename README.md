# Heart Disease Risk Detector 🫀
 
A logistic regression model built **from scratch** using NumPy to predict the risk of heart disease — no scikit-learn, no shortcuts.
The model is deployed using **Streamlit**, allowing anyone to use it through a clean web interface.
 
---
# 🚀 Live Demo


## Overview
 
This project implements a complete binary classification pipeline on the Cleveland Heart Disease Dataset. Every core component — sigmoid function, cost function, gradient descent, Z-score normalization — is written manually in Python to demonstrate a deep understanding of the underlying mathematics.
 
**Model Accuracy: ~84.85%**
 
---
 
## Features Used
 
| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex (0 = Female, 1 = Male) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia type |
 
**Target:** `condition` — 0 (No Disease) or 1 (Disease)
 
---
 
## ⚙️ How It Works

## 1. Data Loading

The Cleveland Heart Disease Dataset is loaded using **Pandas**. The dataset contains 12 clinical features used to predict whether a patient has heart disease.

---

## 2. Feature Normalization

Before training, every feature is standardized using **Z-score Normalization** so that all features are on a similar scale.

```text
z = (x - μ) / σ
```

Where:

- **x** = original feature value
- **μ** = mean of the feature
- **σ** = standard deviation of the feature

Normalization improves the convergence speed of Gradient Descent and prevents features with larger values from dominating the learning process.

---

## 3. Logistic Regression

The model computes a weighted sum of all input features and passes it through the **Sigmoid Function** to produce a probability between 0 and 1.

```text
σ(z) = 1 / (1 + e^(-z))
```

If the probability is:

- **≥ 0.5 → Heart Disease**
- **< 0.5 → No Heart Disease**

---

## 4. Cost Function

The model minimizes the **Binary Cross-Entropy Loss** with **L2 Regularization** to reduce overfitting.

```text
Cost =
-(1/m) Σ [ y log(f(x)) + (1-y) log(1-f(x)) ]
+ (λ / 2m) Σ w²
```

Where:

- **m** = number of training examples
- **λ** = regularization parameter
- **w** = model weights

---

## 5. Gradient Descent

The model learns by repeatedly updating its parameters in the direction that reduces the cost function.

```text
w = w - α × ∂J/∂w
```

```text
b = b - α × ∂J/∂b
```

Where:

- **α** = learning rate
- **J** = cost function

The optimization continues for **1000 iterations** until the model converges.

---


## 6. Web Prediction Flow



User enters health data

        │

        ▼

Feature Normalization

        │

        ▼

Logistic Regression Model

        │

        ▼

Probability Calculation

        │

        ▼

Heart Disease Prediction
 
---
 
## Hyperparameters
 
| Parameter | Value |
|---|---|
| Learning rate (α) | 0.3 |
| Iterations | 1000 |
| Regularization (λ) | 1 |
 
---
 
## Results
 
- **Accuracy:** ~84.85% on training data
- The model correctly classifies heart disease risk based on 12 clinical features
---
 
## Getting Started
 
### Prerequisites
 
```bash
pip install numpy pandas matplotlib
```
 
### Dataset
 
Download the Cleveland Heart Disease dataset:
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
 
Place the CSV file in your working directory and update the path in the notebook:
```python
data_in = pd.read_csv("heart_cleveland_upload.csv")
```
 
### Run the Notebook
 
```bash
jupyter notebook Heart_risk_detector.ipynb
```
 
Run all cells. The last cell will prompt you to enter your health data interactively and predict your heart disease risk.
 
---
 
## Project Structure
 
```
heart-risk-detector/
│
├── Heart_risk_detector.ipynb   # Main notebook
├── heart_cleveland_upload.csv  # Dataset (download separately)
└── README.md
```
 
---

## Output

<img width="842" height="212" alt="image" src="https://github.com/user-attachments/assets/f4579c5c-420c-4482-9cde-c588b3d1cb88" />

<img width="842" height="328" alt="image" src="https://github.com/user-attachments/assets/b367c9c4-364a-4b09-876c-20971f45a36c" />

<img width="430" height="253" alt="image" src="https://github.com/user-attachments/assets/9def0bd6-3ecd-4177-a5ba-5f2d683548d3" />

 
## What I Learned
 
- Implementing logistic regression from scratch using NumPy
- Understanding the mathematics behind sigmoid, cost functions, and gradient descent
- Applying Z-score normalization for feature scaling
- Using L2 regularization to reduce overfitting
- Building an end-to-end ML pipeline without high-level libraries
---
 
## Dataset Credit
 
**Cleveland Heart Disease Dataset**
Source: UCI Machine Learning Repository
Creators: Hungarian Institute of Cardiology, University Hospitals, VA Medical Center, Cleveland Clinic Foundation
 
---
 
## Author
 
**Varad** — CS Student | Machine Learning Enthusiast | Python Developer
 
*Built as part of a self-directed ML learning journey alongside the Andrew Ng Machine Learning Specialization.*
