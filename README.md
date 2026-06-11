# Comprehensive Machine Learning Portfolio: Frameworks, Foundations & Deep Learning

Welcome to my end-to-end Machine Learning portfolio repository. This repository contains a structured, production-grade curriculum tracking across three progressive execution tiers: from core data preprocessing and classic linear baselines up to non-linear ensemble frameworks, geometric hyperplanes, and deep learning architectures. 

The implementations focus on rigorous engineering standards: avoiding data leakage via split-first mechanics, utilizing object-oriented pipeline architectures, and generating rich mathematical visual diagnostics.

---

## 🛠️ Repository Architecture & Frameworks
* **Language:** Python 3.12+
* **Core Ecosystem:** `scikit-learn`, `pandas`, `numpy`
* **Deep Learning Engine:** `TensorFlow` / `Keras`
* **Data Visualization:** `matplotlib`, `seaborn`

---

## 📈 Executive Summary of Project Tasks

### 📊 Level 1: Foundations & Supervised Core

#### Task 1: Data Preprocessing Pipeline
* **Objective:** Establish a repeatable, data-leakage-proof workflow to handle messy raw datasets containing text categories and missing data fields.
* **Engineering Strategy:** Leveraged scikit-learn's `ColumnTransformer` to isolate pipelines. Numerical columns are handled via `SimpleImputer(strategy='median')` and centered using `StandardScaler()`. Categorical features are dynamically expanded via `OneHotEncoder(handle_unknown='ignore')`. 
* **Key Metric Verification:** Successfully scaled datasets to a target **Mean $\approx 0.0$** and **Standard Deviation $= 1.0$** with $0$ remaining null values.

#### Task 2: Simple Linear Regression
* **Objective:** Model a continuous trend line predicting closing asset variations against day opening metrics.
* **Mathematical Baseline:** Built using Ordinary Least Squares (OLS) targeting $y = mX + c$.
* **Performance Matrix:**
  * **Slope ($m$):** $0.9999$
  * **Intercept ($c$):** $0.0267$
  * **Mean Squared Error (MSE):** $2.7261$
  * **Variance Explained ($R^2$):** **$99.97\%$**
* **Diagnostic Visualization:** Generates a high-precision scatter trend overlay plot demonstrating tight residuals across historical boundaries.

#### Task 3: K-Nearest Neighbors (KNN) Classifier
* **Objective:** Classify points geometrically by mapping majority voting matrices within local Euclidean space.
* **Engineering Strategy:** Implemented an odd-integer hyperparameter sweep ($K = [1, 3, 5, 7, 9, 15]$) evaluating the holdout test set to pinpoint the optimal neighborhood while bypassing local outlier noise.

---

### 🧠 Level 2: Advanced Classifiers & Clustering

#### Task 1: Logistic Regression for Churn Prediction
* **Objective:** Model binary class risks while adjusting for heavily skewed class imbalances.
* **Engineering Strategy:** Trained using `class_weight='balanced'` to enforce structural sensitivity thresholds over minority indicators. Raw log-odds are systematically converted into executive-friendly **Odds Ratios ($e^{\text{coefficient}}$)**.
* **Key Insights Uncovered:** * `International plan_Yes` acts as a heavy accelerator (**$3.18\times$ higher odds** of churn).
  * `Customer service calls` increases churn odds by **$2.70\times$ per call**.
* **Model Discrimination:** Achieved an **ROC-AUC Score of $0.8039$**, proving strong discriminative capabilities.

#### Task 2: Structural Decision Trees
* **Objective:** Map structural, human-interpretable logic branching flows for multi-class floral categorization.
* **Engineering Strategy:** Deployed a pre-pruned `DecisionTreeClassifier(max_depth=3)` using Gini Impurity splits.
* **Performance Scorecard:** Achieved an exceptional **$97.78\%$ Test Accuracy**, misclassifying only a single sample out of the holdout data pool.
* **Diagnostic Visualization:** Generates a clear graphical branching flow map tracking node metrics from the root split (`petal_length <= 2.45`) down to pure leaf boxes.

#### Task 3: Unsupervised K-Means Clustering
* **Objective:** Uncover latent structural groupings within unlabelled coordinate points.
* **Engineering Strategy:** Utilized the **Elbow Method** (calculating Within-Cluster Sum of Squares / Inertia over a 10-epoch iteration sweep) to dynamically determine the true physical cluster counts, plotting the resulting data partitions alongside exact calculated centroid maps.

---

### 🚀 Level 3: Ensembles, Hyperplanes & Deep Learning

#### Task 1: Random Forest Ensembles (Feature Importance Tracking)
* **Objective:** Mitigate single-tree overfitting by training a bag of decorrelated structural trees.
* **Key Metric Verification:** Evaluated out-of-sample data using strict Precision, Recall, and F1-score balances. 
* **Diagnostic Visualization:** Renders a feature importance ranking distribution chart proving that daytime usage volume handles over 12% of the model's split decisions.

#### Task 2: Support Vector Machines (SVM) Decision Projections
* **Objective:** Maximize geometric margin separation lines using alternate dimensional hyperplanes.
* **Engineering Strategy:** Contrasted a **Linear Kernel** against a non-linear **Radial Basis Function (RBF) Kernel**.
* **Diagnostic Visualization:** Generates side-by-side spatial boundary decision maps, illustrating how the RBF configuration curves to wrap around overlapping boundaries where linear models fail.

#### Task 3: Deep Learning (Multi-Layer Perceptron Neural Network)
* **Objective:** Implement a dense, non-linear deep neural architecture via TensorFlow.
* **Network Blueprint:** Structured with input layer sizing, hidden layers bound to `ReLU` activation functions, custom `Dropout(0.2)` regularization vectors to prevent network overfitting, and a terminal `Softmax` array layer.
* **Diagnostic Visualization:** Outputs side-by-side validation loss trajectories, showing smooth, parallel declines down to **$97.74\%$ testing accuracy** without any overfitting divergences.

---

## 🚀 How to Clone and Execute This Repository

### 1. Replicate the Environment
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/ml-portfolio-curriculum.git](https://github.com/YOUR_USERNAME/ml-portfolio-curriculum.git)
cd ml-portfolio-curriculum

# Install dependencies
pip install -r requirements.txt
