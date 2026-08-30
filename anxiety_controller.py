
# Anxiety prediction controller for Webots
# Uses the SAME 12 input features identified in the user's ML analysis.
# Keys:
#   1 = Low/Minimal demo profile
#   2 = Mild demo profile
#   3 = Moderate demo profile
#   4 = Severe demo profile
#   R = reload model
#   Q = quit
#
# Required exported files:
#   federated_anxiety_model.keras
#   preprocessor.pkl
#   label_encoder.pkl
#   feature_metadata.json
#
# Put these files in this controller folder.

from controller import Robot, Keyboard
import os
import json
import csv
import numpy as np

try:
    import tensorflow as tf
    import joblib
except Exception as e:
    tf = None
    joblib = None
    print("WARNING: TensorFlow/joblib could not be imported:", e)

TIME_STEP = 32
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "federated_anxiety_model.keras")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
METADATA_FILE = os.path.join(BASE_DIR, "feature_metadata.json")
LOG_FILE = os.path.join(BASE_DIR, "prediction_log.csv")

FEATURES = [
    "Age",
    "Gender",
    "CGPA",
    "Academic_Pressure",
    "Financial_Stress",
    "Study_Hours_Per_Day",
    "Sleep_Duration",
    "Social_Support",
    "Exercise_Days_Per_Week",
    "Diet_Quality",
    "Family_History_Mental_Illness",
    "Perceived_Stress"
]

# These demo profiles are ONLY for testing the Webots interface.
# The actual prediction comes from your exported federated model.
PROFILES = {
    1: {
        "Age": 21, "Gender": "Female", "CGPA": 3.5,
        "Academic_Pressure": 1, "Financial_Stress": 1,
        "Study_Hours_Per_Day": 4, "Sleep_Duration": 8,
        "Social_Support": 4, "Exercise_Days_Per_Week": 4,
        "Diet_Quality": 4, "Family_History_Mental_Illness": "No",
        "Perceived_Stress": 1
    },
    2: {
        "Age": 21, "Gender": "Female", "CGPA": 3.2,
        "Academic_Pressure": 2, "Financial_Stress": 2,
        "Study_Hours_Per_Day": 6, "Sleep_Duration": 7,
        "Social_Support": 3, "Exercise_Days_Per_Week": 3,
        "Diet_Quality": 3, "Family_History_Mental_Illness": "No",
        "Perceived_Stress": 2
    },
    3: {
        "Age": 22, "Gender": "Female", "CGPA": 2.8,
        "Academic_Pressure": 4, "Financial_Stress": 3,
        "Study_Hours_Per_Day": 8, "Sleep_Duration": 5,
        "Social_Support": 2, "Exercise_Days_Per_Week": 1,
        "Diet_Quality": 2, "Family_History_Mental_Illness": "Yes",
        "Perceived_Stress": 4
    },
    4: {
        "Age": 23, "Gender": "Male", "CGPA": 2.4,
        "Academic_Pressure": 5, "Financial_Stress": 5,
        "Study_Hours_Per_Day": 10, "Sleep_Duration": 4,
        "Social_Support": 1, "Exercise_Days_Per_Week": 0,
        "Diet_Quality": 1, "Family_History_Mental_Illness": "Yes",
        "Perceived_Stress": 5
    }
}

robot = Robot()
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

# Optional motors: if the names exist in the PROTO, they will be used.
motors = []
for name in ["head_yaw_motor", "head_pitch_motor", "antenna_left_motor", "antenna_right_motor"]:
    try:
        m = robot.getDevice(name)
        if m:
            motors.append((name, m))
    except:
        pass

def move_robot(level):
    # Simple expressive motion. This is not a clinical action.
    try:
        for name, motor in motors:
            if "head_yaw" in name:
                motor.setPosition(0.15 if level in ["Moderate Anxiety", "Severe Anxiety"] else 0.0)
            elif "head_pitch" in name:
                motor.setPosition(-0.10 if level == "Severe Anxiety" else 0.0)
            elif "antenna_left" in name:
                motor.setPosition(0.25 if level in ["Mild Anxiety", "Moderate Anxiety"] else 0.0)
            elif "antenna_right" in name:
                motor.setPosition(-0.25 if level in ["Mild Anxiety", "Moderate Anxiety"] else 0.0)
    except Exception:
        pass

def load_assets():
    if tf is None or joblib is None:
        print("\nTensorFlow/joblib is unavailable in this controller environment.")
        return None, None, None

    missing = [f for f in [MODEL_FILE, SCALER_FILE] if not os.path.exists(f)]
    if missing:
        print("\nMODEL FILES NOT FOUND:")
        for f in missing:
            print("  ", f)
        print("\nFirst export the trained model from Kaggle, then copy the files here.")
        return None, None, None

    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        preprocessor = joblib.load(SCALER_FILE)
        label_encoder = None
        print("\nFederated model + scaler loaded successfully.")
        return model, preprocessor, label_encoder
    except Exception as e:
        print("\nCould not load deployment files:", e)
        return None, None, None

model, preprocessor, label_encoder = load_assets()

def prepare_input(profile):
    # Build a one-row DataFrame in the exact feature order.
    import pandas as pd
    row = {f: profile[f] for f in FEATURES}
    df = pd.DataFrame([row], columns=FEATURES)

    # Apply the exact StandardScaler exported from the training notebook.
    X = preprocessor.transform(df)
    return np.asarray(X, dtype=np.float32)

def log_prediction(profile_id, profile, pred_class, probs):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Profile", *FEATURES, "Predicted_Level", "Confidence"])
        writer.writerow([
            profile_id,
            *[profile[x] for x in FEATURES],
            pred_class,
            float(np.max(probs))
        ])

def predict(profile_id):
    global model, preprocessor, label_encoder

    print("\n" + "=" * 60)
    print("ANXIETY RISK PREDICTION - WEBOTS")
    print("=" * 60)

    profile = PROFILES[profile_id]

    print("\nInput profile:")
    for f in FEATURES:
        print(f"  {f}: {profile[f]}")

    if model is None:
        print("\nNo trained model is loaded.")
        print("Webots interface is ready, but prediction requires the exported")
        print("federated model + preprocessing files.")
        return

    try:
        X = prepare_input(profile)
        probs = np.asarray(model.predict(X, verbose=0)).reshape(-1)

        risk_probability = float(probs[0])
        pred_text = "High Anxiety Risk" if risk_probability >= 0.5 else "Low Anxiety Risk"
        print("\\nPrediction:", pred_text)
        print(f"High-risk probability: {risk_probability:.4f}")
        print(f"Low-risk probability : {1.0-risk_probability:.4f}")

        move_robot(pred_text)
        log_prediction(profile_id, profile, pred_text, probs)

        print("\nRobot response:")
        print("  Prediction is a screening/risk estimate, not a diagnosis.")
        print("=" * 60)

    except Exception as e:
        print("\nPrediction error:", e)
        print("Check that the exported preprocessor matches the trained model.")

print("\n" + "=" * 60)
print("REACHY-MINI-STYLE ANXIETY DEMO")
print("=" * 60)
print("1 = Minimal/low-risk demo profile")
print("2 = Mild-risk demo profile")
print("3 = Moderate-risk demo profile")
print("4 = Severe-risk demo profile")
print("R = Reload model")
print("Q = Quit")
print("=" * 60)

while robot.step(TIME_STEP) != -1:
    key = keyboard.getKey()

    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        predict(int(chr(key)))

    elif key in [ord('R'), ord('r')]:
        model, preprocessor, label_encoder = load_assets()

    elif key in [ord('Q'), ord('q')]:
        break
