# Fraud Detection in Transactions
# Using Logistic Regression and Random Forest with Isotonic Calibration and TimeSeriesSplit
# Target column: "Class"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

import warnings
warnings.filterwarnings('ignore')


#  Load Dataset

df = pd.read_csv('/Users/akhilduggal/Downloads/creditcard.csv')
print("Dataset head:\n", df.head())
print("\nDataset info:")
print(df.info())


#  EDA

print("\nMissing values:\n", df.isnull().sum())

# Target distribution
plt.figure(figsize=(8,6))
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Non-Fraud Transactions')
plt.tight_layout()
plt.savefig('plot1_target_distribution.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot2_correlation_heatmap.png')
plt.close()


#  Preprocessing

df.fillna(df.median(), inplace=True)
df = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()
X = df.drop('Class', axis=1)
X_scaled = scaler.fit_transform(X)
y = df['Class']


#  TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)


#  Model Training with Calibration

lr = LogisticRegression()
lr_calibrated = CalibratedClassifierCV(lr, method='isotonic', cv=tscv)
lr_calibrated.fit(X_scaled, y)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=tscv)
rf_calibrated.fit(X_scaled, y)


#  Model Evaluation

models = {'Logistic Regression': lr_calibrated,
          'Random Forest': rf_calibrated}

for name, model in models.items():
    y_prob = model.predict_proba(X_scaled)[:,1]
    y_pred = model.predict(X_scaled)
    
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("ROC-AUC:", roc_auc_score(y, y_prob))
    print("Brier Score (Calibration):", brier_score_loss(y, y_prob))
    print(classification_report(y, y_pred))


# Calibration Plots

plt.figure(figsize=(10,6))
for name, model in models.items():
    prob_true, prob_pred = calibration_curve(y, model.predict_proba(X_scaled)[:,1], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)

plt.plot([0,1],[0,1],'k--')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Plots')
plt.legend()
plt.tight_layout()
plt.savefig('plot3_calibration.png')
plt.close()

