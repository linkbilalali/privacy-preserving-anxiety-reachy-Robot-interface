# Anxiety Robot — Privacy-Preserving Explainable AI Framework

A Python-based anxiety robot interfacde animate research project for predicting anxiety risk among university students using machine learning, Federated Learning with FedAvg, explainable AI, and a Reachy-inspired robot simulation.

> **Important:** The robot component is a Python-based visual simulation/interface. It is **not** an official physical Reachy Mini implementation and is **not** a Webots Reachy Mini simulation. The system provides an AI-based anxiety-risk indication and should not be treated as a clinical diagnosis.

---

## Project Overview

The project follows this overall pipeline:

```text
Student Anxiety Dataset
        ↓
Data Cleaning & Preprocessing
        ↓
Centralized Machine Learning Analysis
        ↓
Federated Learning using FedAvg
        ↓
Global Federated Anxiety Model
        ↓
Prediction + Explainable AI
        ↓
Anxiety Risk Score
        ↓
LOW / MODERATE / HIGH
        ↓
Python Anxiety Robot Simulation
```

---

## Dataset

The project uses the **Anxiety Test Data** dataset stored as:

```text
data/Anxiety.csv
```

The dataset contains:

- **2,028 student records**
- **16 columns**
- Demographic information
- Academic information
- Scholarship/waiver information
- Seven academic-pressure/anxiety questionnaire items
- Anxiety Value
- Anxiety Label

### Dataset variables

| Variable | Description |
|---|---|
| Age | Student age group |
| Gender | Student gender |
| University | University attended |
| Department | Academic department |
| Academic Year | Current academic year |
| Current CGPA | CGPA range |
| Waiver/Scholarship | Scholarship or fee-waiver status |
| Q1–Q7 | Seven anxiety-related questionnaire responses |
| Anxiety Value | Overall anxiety score |
| Anxiety Label | Anxiety severity class |

The questionnaire responses are ordinal values ranging from **0 to 3**. The dataset's Anxiety Value ranges from **0 to 21**.

---

## Notebook Structure

### Notebook 1 — Data Preprocessing and Centralized Model

```text
notebooks/01_data_preprocessing_centralized.ipynb
```

Main activities:

1. Load the Anxiety dataset
2. Inspect the dataset
3. Check missing values
4. Remove duplicate records
5. Perform exploratory data analysis
6. Encode categorical variables
7. Prepare model features
8. Exclude the derived `Anxiety Value` from the predictive features to reduce target leakage
9. Train/evaluate the centralized model
10. Generate classification and evaluation results

The notebook reports:

- Original dataset: **2,028 × 16**
- Duplicate rows identified: **65**
- Dataset after cleaning: **1,963 × 16**

Centralized model results:

- Accuracy: **93.00%**
- Macro Precision: **94.00%**
- Macro Recall: **92.00%**
- Macro F1-Score: **93.00%**
- Weighted F1-Score: **93.00%**
- ROC-AUC: **99.22%**

---

### Notebook 2 & 3 — Federated Learning with FedAvg

```text
notebooks/02_federated_learning_fedavg.ipynb
```

The federated-learning experiment simulates multiple clients and performs local training followed by FedAvg aggregation.

Configuration used in the notebook:

- **5 simulated clients**
- **5 communication rounds**
- **15 encoded input features**
- **4 anxiety classes**
- FedAvg aggregation

Round-wise global performance:

| Round | Accuracy | Loss |
|---:|---:|---:|
| 1 | 93.35% | 0.1891 |
| 2 | 95.32% | 0.1175 |
| 3 | 97.78% | 0.0810 |
| 4 | 98.03% | 0.0709 |
| 5 | 97.78% | 0.0651 |

Final federated model performance:

| Metric | Result |
|---|---:|
| Accuracy | **97.78%** |
| Precision | **97.99%** |
| Recall | **97.78%** |
| F1-Score | **97.82%** |
| ROC-AUC | **99.94%** |

The final trained model is saved as:

```text
models/federated_anxiety_model.keras
```

---

## Anxiety Classes

The federated model uses four classes:

```text
Class 0 → Minimal Anxiety
Class 1 → Mild Anxiety
Class 2 → Moderate Anxiety
Class 3 → Severe Anxiety
```

For the robot interface, these four model classes are converted into three visual robot states:

```text
Minimal / Mild     → LOW
Moderate           → MODERATE
Severe             → HIGH
```

The robot anxiety score is calculated from the predicted class probabilities.

---

## Explainable AI

The project includes explainability analysis using:

- **SHAP**
- **LIME**

These methods are used to investigate feature contributions and improve the interpretability of model predictions.

The project also contains a LIME-style feature importance visualization:

```text
results/lime_feature_importance.png
```

---

## Final Anxiety Robot Simulation

### Notebook 4

```text
notebooks/03_anxiety_robot_final.ipynb
```

The final notebook does **not retrain the federated model**. It loads the trained model and associated preprocessing assets and applies them to the anxiety test data.

Main workflow:

```text
Anxiety.csv
    ↓
Feature Preparation
    ↓
Scaler
    ↓
federated_anxiety_model.keras
    ↓
4-Class Prediction
    ↓
Anxiety Score
    ↓
Robot Level
    ↓
Animated Robot
```

### Robot states

#### LOW

- Green eyes
- Low anxiety indication
- Low anxiety score
- Simple hand movement
- Head movement

#### MODERATE

- Yellow eyes
- Moderate anxiety indication
- Moderate anxiety score
- Simple hand movement
- Head movement

#### HIGH

- Red eyes
- High anxiety indication
- High anxiety score
- Simple hand movement
- Head movement

The robot animation is intentionally kept simple for demonstration and research presentation.

---

## Final Project Files

```text
ANXIETY_ROBOT/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── Anxiety.csv
│
├── models/
│   ├── federated_anxiety_model.keras
│   ├── scaler.pkl
│   └── feature_metadata.json
│
├── notebooks/
│   ├── 01_data_preprocessing_centralized.ipynb
│   ├── 02_federated_learning_fedavg.ipynb
│   └── 03_anxiety_robot_final.ipynb
│
├── robot/
│   └── anxiety_controller.py
│
├── results/
│   ├── anxiety_robot_final_results.csv
│   └── lime_feature_importance.png
│
└── demo/
    └── anxiety_robot_demo.gif
```

---

## Model and Preprocessing Assets

### `federated_anxiety_model.keras`

The final trained global Federated Learning model used by the robot prediction pipeline.

### `scaler.pkl`

The preprocessing scaler used to transform model input features consistently.

### `feature_metadata.json`

Stores feature-related metadata required for consistent model input and integration.

### `anxiety_controller.py`

Contains anxiety prediction/controller logic used as part of the project integration.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main implementation language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Preprocessing and evaluation |
| TensorFlow / Keras | Neural-network model handling |
| Matplotlib | Visualisation and robot animation |
| Seaborn | Statistical visualisation |
| SHAP | Explainable AI |
| LIME | Local model explanations |
| Pickle | Scaler serialization/loading |
| JSON | Feature metadata |
| Jupyter / Kaggle Notebook | Experimental environment |

---

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

For notebook execution, Jupyter Notebook or JupyterLab can be used.

---

## Running the Project

### 1. Prepare the dataset

Place the dataset at:

```text
data/Anxiety.csv
```

### 2. Run Notebook 1

Run:

```text
01_data_preprocessing_centralized.ipynb
```

This performs data preparation and centralized model analysis.

### 3. Run Notebook 2 & 3

Run:

```text
02_federated_learning_fedavg.ipynb
```

This performs the simulated Federated Learning experiment and saves the global model.

### 4. Run the final robot notebook

Run:

```text
03_anxiety_robot_final.ipynb
```

Ensure the following assets are available:

```text
federated_anxiety_model.keras
scaler.pkl
feature_metadata.json
anxiety_controller.py
Anxiety.csv
```

The notebook generates anxiety predictions, anxiety scores, robot levels, and the animated robot demonstration.

---

## Results Summary

| Experiment | Accuracy | F1-Score | ROC-AUC |
|---|---:|---:|---:|
| Centralized Model | **93.00%** | **93.00%** | **99.22%** |
| Federated Model | **97.78%** | **97.82%** | **99.94%** |

The federated experiment achieved higher reported predictive performance than the centralized experiment on their respective evaluation configurations. The federated model also achieved a ROC-AUC of **99.94%**.

---

## Research Contribution

The project combines:

1. Student anxiety-risk prediction
2. Centralized machine-learning evaluation
3. Federated Learning using FedAvg
4. Explainable AI using SHAP and LIME
5. Consistent preprocessing through retained model assets
6. A Python-based anxiety robot interface
7. Visual LOW, MODERATE, and HIGH anxiety responses

The robot interface provides a visual demonstration layer on top of the trained anxiety prediction pipeline.

---

## Disclaimer

This project is intended for **research and educational purposes**. The model output represents an AI-based anxiety-risk indication and should not be interpreted as a medical or clinical diagnosis. Appropriate professional assessment should be used for clinical decisions.

---

## Author

**Anxiety Robot Research Project**

This repository contains the implementation and experimental artifacts associated with the anxiety-risk prediction and robotic-interface research project.
