import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# Step 1: Load Training Dataset
# ---------------------------------------------------------
df = pd.read_csv("import numpy as np")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# Step 1: Load Training Dataset
# ---------------------------------------------------------
df = pd.read_csv("/content/engine_static_fire_train.csv")

# Use the correct column names from your dataset
X = df[['Chamber_Pressure_F',
        'LOX_Mass_Flow_kg',
        'Fuel_Mass_Flow_kg',
        'Turbopump_RPM',
        'Thrust_kN',
        'Chamber_Temp_K']]

y = df['Abort']   # label column

# ---------------------------------------------------------
# Step 2: Train Predictive Model
# ---------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_val)
print("Model Performance Report:")
print(classification_report(y_val, predictions, target_names=['Nominal (0)', 'Abort (1)']))

# ---------------------------------------------------------
# Step 3: Load Testing Dataset (Telemetry Simulation)
# ---------------------------------------------------------
test_df = pd.read_csv("/content/engine_static_fire_train_v2.csv")

for idx, row in test_df.iterrows():
    current_features = row[['Chamber_Pressure_F',
                            'LOX_Mass_Flow_kg',
                            'Fuel_Mass_Flow_kg',
                            'Turbopump_RPM',
                            'Thrust_kN',
                            'Chamber_Temp_K']].values.reshape(1, -1)

    predicted_state = model.predict(current_features)[0]
    abort_probability = model.predict_proba(current_features)[0][1]

    print("-" * 40)
    print(f"Telemetry Sample {idx}:")
    if predicted_state == 1:
        print(f"🚨 ABORT TRIGGERED! (Confidence: {abort_probability*100:.1f}%)")
    else:
        print(f"✅ NOMINAL. (Risk of abort: {abort_probability*100:.1f}%)")

# ---------------------------------------------------------
# Step 4: Feature Importance Visualization
# ---------------------------------------------------------
importances = model.feature_importances_
plt.barh(X.columns, importances, color='teal')
plt.xlabel("Importance in Triggering Abort")
plt.title("Telemetry Feature Importance")
plt.show()
")

# Use the correct column names from your dataset
X = df[['Chamber_Pressure_F',
        'LOX_Mass_Flow_kg',
        'Fuel_Mass_Flow_kg',
        'Turbopump_RPM',
        'Thrust_kN',
        'Chamber_Temp_K']]

y = df['Abort']   # label column

# ---------------------------------------------------------
# Step 2: Train Predictive Model
# ---------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_val)
print("Model Performance Report:")
print(classification_report(y_val, predictions, target_names=['Nominal (0)', 'Abort (1)']))

# ---------------------------------------------------------
# Step 3: Load Testing Dataset (Telemetry Simulation)
# ---------------------------------------------------------
test_df = pd.read_csv("/content/engine_static_fire_test.csv")

for idx, row in test_df.iterrows():
    current_features = row[['Chamber_Pressure_F',
                            'LOX_Mass_Flow_kg',
                            'Fuel_Mass_Flow_kg',
                            'Turbopump_RPM',
                            'Thrust_kN',
                            'Chamber_Temp_K']].values.reshape(1, -1)

    predicted_state = model.predict(current_features)[0]
    abort_probability = model.predict_proba(current_features)[0][1]

    print("-" * 40)
    print(f"Telemetry Sample {idx}:")
    if predicted_state == 1:
        print(f"🚨 ABORT TRIGGERED! (Confidence: {abort_probability*100:.1f}%)")
    else:
        print(f"✅ NOMINAL. (Risk of abort: {abort_probability*100:.1f}%)")

# ---------------------------------------------------------
# Step 4: Feature Importance Visualization
# ---------------------------------------------------------
importances = model.feature_importances_
plt.barh(X.columns, importances, color='teal')
plt.xlabel("Importance in Triggering Abort")
plt.title("Telemetry Feature Importance")
plt.show()
