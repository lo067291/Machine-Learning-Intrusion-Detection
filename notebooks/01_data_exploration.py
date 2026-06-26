import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Column names for NSL-KDD dataset

columns = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

#Load the dataset using direct file path
df = pd.read_csv(r'C:\Users\stacy\PycharmProjects\PythonProjects\Machine-Learning-Intrusion-Detection\data\KDDTrain+.txt', names=columns)

#Basic Info
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLabel distribution:")
print(df['label'].value_counts())

#check for unique values in categorical columns
print("\nUnique protocol types:", df['protocol_type'].unique())
print("Unique services:", df['service'].nunique())
print("Unique flags:", df['flag'].unique())

#Create Encoder
le = LabelEncoder()

#Convert text categories to numbers
df['protocol_type'] = le.fit_transform(df['protocol_type'])
df['service'] = le.fit_transform(df['service'])
df['flag'] = le.fit_transform(df['flag'])

print("\nAfter encoding:")
print(df[['protocol_type', 'service', 'flag']].head())

#Make Binary Classification

df['is_attack'] = df['label'].apply(lambda x: 0 if x.strip() == 'normal' else 1)

print("\nBinary label distribution:")
print(df['is_attack'].value_counts())
print(f"\nNormal traffic: {(df['is_attack']==0).sum()} samples")
print(f"Attack traffic: {(df['is_attack']==1).sum()} samples")
print(f"\nAttack rate: {(df['is_attack']==1).mean()*100:.1f}%")

#Seperate features (X) and target label (Y)
X = df.drop(['label', 'difficulty', 'is_attack'], axis = 1)
y = df['is_attack']

#Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}")
print(f"\nTraining attack rate: {y_train.mean()*100:.1f}%")
print(f"Testing attack rate: {y_test.mean()*100:.1f}%")


#strat training the first model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

print("Training Logistic Regression...")
start = time.time()

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

training_time = time.time() - start

# Make predictions
lr_predictions = lr_model.predict(X_test_scaled)

# Evaluate
print(f"\nTraining time: {training_time:.2f} seconds")
print(f"\nAccuracy: {accuracy_score(y_test, lr_predictions)*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, lr_predictions, target_names=['Normal', 'Attack']))



#Training random forest now
from sklearn.ensemble import RandomForestClassifier

print("Training Random Forest...")
start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_time = time.time() - start

# Predictions (Random Forest uses unscaled data)
rf_predictions = rf_model.predict(X_test)

print(f"\nTraining time: {rf_time:.2f} seconds")
print(f"\nAccuracy: {accuracy_score(y_test, rf_predictions)*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, rf_predictions, target_names=['Normal', 'Attack']))


#Now train the neural network
from sklearn.neural_network import MLPClassifier

print("Training Neural Network...")
start = time.time()

nn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=100,
    random_state=42,
    verbose=False
)
nn_model.fit(X_train_scaled, y_train)

nn_time = time.time() - start

# Predictions
nn_predictions = nn_model.predict(X_test_scaled)

print(f"\nTraining time: {nn_time:.2f} seconds")
print(f"\nAccuracy: {accuracy_score(y_test, nn_predictions)*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, nn_predictions, target_names=['Normal', 'Attack']))


#This will make a graph to visualize the results

import os
os.makedirs('visualizations', exist_ok=True)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create figure with 3 confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

models = [
    (lr_predictions, 'Logistic Regression\n95.27%'),
    (rf_predictions, 'Random Forest\n99.86%'),
    (nn_predictions, 'Neural Network\n99.61%')
]

for ax, (predictions, title) in zip(axes, models):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=['Normal', 'Attack'],
        yticklabels=['Normal', 'Attack']
    )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.suptitle('Network Intrusion Detection - Model Comparison',fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('visualizations/confusion_matrices.png',dpi=150, bbox_inches='tight')
plt.show()
print("Saved to visualizations/confusion_matrices.png")

#Feature importance from Random Forest, this determines which attack random forest relies on to determine if it is a attack or not
    #Tells you what attackers can't easily hide
    #Helps you understand WHY certain traffic looks malicious
    #Real security insight, not just a number


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=feature_importance.head(15),
    x='importance',
    y='feature',
    palette='Blues_r'
)
plt.title('Top 15 Most Important Features\nfor Network Intrusion Detection', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nTop 5 most important features:")
print(feature_importance.head())



#ROC curve comparisons, how does my model perform at different detection levels
    #change the threshold from >.5 to >.3
    #how will it deal with the increased false alarms


from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 6))

models_for_roc = [
    (lr_model, X_test_scaled, 'Logistic Regression', 'blue'),
    (rf_model, X_test, 'Random Forest', 'green'),
    (nn_model, X_test_scaled, 'Neural Network', 'red')
]

for model, X_t, name, color in models_for_roc:
    # Get probability scores
    proba = model.predict_proba(X_t)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')

# Random baseline
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5)')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve - Network Intrusion Detection\nModel Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()


#Save the Models to joblib

import joblib
import os

os.makedirs('models', exist_ok=True)

# Save all three models and the scaler
joblib.dump(lr_model, 'models/logistic_regression.pkl')
joblib.dump(rf_model, 'models/random_forest.pkl')
joblib.dump(nn_model, 'models/neural_network.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ All models saved to models/ folder")