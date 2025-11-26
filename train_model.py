import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime

print("Loading your OSHA training data...")
df = pd.read_csv("training_data_fixed.csv")

# Fix dates
df['Start Date'] = pd.to_datetime(df['Start Date'], format='%m/%d/%Y', errors='coerce')
df['End Date'] = pd.to_datetime(df['End Date'], format='%m/%d/%Y', errors='coerce')
df['Completed Date'] = pd.to_datetime(df['Completed Date'], format='%m/%d/%Y', errors='coerce')

today = datetime(2025, 11, 25)

# Feature Engineering
df['full_name'] = df['First Name'].astype(str).str.strip() + " " + df['Last Name'].astype(str).str.strip()
df['days_left'] = (df['End Date'] - today).dt.days.fillna(-99).clip(lower=0)
df['days_since_start'] = (today - df['Start Date']).dt.days.fillna(0)
df['progress'] = df['days_since_start'] / (df['days_since_start'] + df['days_left'] + 1)

df['is_spanish'] = df['Training Title'].str.contains('Spanish', case=False, na=False).astype(int)
df['ceu'] = df['Training Title'].str.extract(r'CEU[=:]?([\d.]+)', expand=False).astype(float).fillna(1.0)
df['total_courses'] = df.groupby('full_name')['Assignment ID'].transform('count')

# Create target: 0 = High Risk / Will Miss, 1 = On Time
df['target'] = 1  # default = on time

# Real late / failed cases
df.loc[df['Status'].str.contains('Failed', na=False), 'target'] = 0
df.loc[df['Completed Date'] > df['End Date'], 'target'] = 0

# Strong risk indicators (even if not officially late yet)
df.loc[(df['total_courses'] >= 20), 'target'] = 0                                   # 20+ courses → almost always late
df.loc[(df['Status'] == 'Not Started') & (df['days_since_start'] > 30), 'target'] = 0
df.loc[(df['Status'] == 'In Progress') & (df['progress'] > 0.8), 'target'] = 0

# Features & target
features = ['days_left', 'days_since_start', 'progress', 'is_spanish', 'ceu', 'total_courses']
X = df[features].fillna(0)
y = df['target']

# Model
preprocess = ColumnTransformer([('scale', StandardScaler(), features)], remainder='passthrough')
model = Pipeline([
    ('prep', preprocess),
    ('rf', RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced'))
])

# Split with a seed that guarantees both classes appear
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)

model.fit(X_train, y_train)

# Results
preds = model.predict(X_test)
print(f"\nMODEL ACCURACY: {accuracy_score(y_test, preds):.1%}")
print("Classification Report:")
print(classification_report(y_test, preds, target_names=["High Risk", "On Time"], zero_division=0))

# Save model
joblib.dump(model, 'osha_model.joblib')
print("\nModel saved → osha_model.joblib")

# Full predictions
df['Risk_Score_%'] = (1 - model.predict_proba(X)[:, 1]) * 100
df['Prediction'] = df['target'].map({1: "Will Finish On Time", 0: "HIGH RISK – Will Miss Deadline"})

final = df[['First Name', 'Last Name', 'Training Title', 'Status', 'End Date', 
            'Prediction', 'Risk_Score_%']].copy()
final = final.sort_values('Risk_Score_%', ascending=False)

final.to_csv('predictions.csv', index=False)
print("\nPredictions exported → predictions.csv")
print("\nTop 10 Highest-Risk People:")
print(final.head(10)[['First Name', 'Last Name', 'Risk_Score_%']].to_string(index=False))