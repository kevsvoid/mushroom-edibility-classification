import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_curve, auc
)

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
df = pd.read_csv('data/mushrooms.csv')

# Fix column names (important)
df.columns = df.columns.str.replace('-', '_')

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['class'].value_counts())

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
le = LabelEncoder()
df_enc = df.copy()

for col in df_enc.columns:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

X = df_enc.drop('class', axis=1)
y = df_enc['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. GRADIENT BOOSTING MODEL
# ─────────────────────────────────────────────
model = GradientBoostingClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"\nGradient Boosting -> Acc={acc:.4f}  AUC={roc_auc:.4f}")

# ─────────────────────────────────────────────
# CREATE OUTPUT FOLDER
# ─────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────
# FIGURE 1 – Dataset Overview
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 1: Dataset Overview', fontweight='bold')

# Class distribution
class_counts = df['class'].map({'e':'Edible','p':'Poisonous'}).value_counts()
axes[0].bar(class_counts.index, class_counts.values)
axes[0].set_title('Class Distribution')

# Correlation heatmap
top_feats = X.columns[:10]
corr = pd.concat([X[top_feats], y], axis=1).corr()
sns.heatmap(corr, ax=axes[1], annot=True, fmt='.2f')

plt.tight_layout()
plt.savefig('outputs/fig1_dataset.png')
plt.close()

# ─────────────────────────────────────────────
# FIGURE 2 – Feature Distributions
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 2: Feature Distributions', fontweight='bold')

# Odor
odor_map = {'a':'almond','c':'creosote','f':'foul','l':'anise',
            'm':'musty','n':'none','p':'pungent','s':'spicy','y':'fishy'}
df['odor_name'] = df['odor'].map(odor_map).fillna(df['odor'])

df.groupby(['odor_name','class']).size().unstack().plot(kind='bar', ax=axes[0])
axes[0].set_title('Odor Distribution')

# Cap color
color_map = {'b':'buff','c':'cinnamon','e':'red','g':'gray','n':'brown',
             'p':'pink','r':'green','u':'purple','w':'white','y':'yellow'}

df['cap_color_name'] = df['cap_color'].map(color_map).fillna(df['cap_color'])
df.groupby(['cap_color_name','class']).size().unstack().plot(kind='bar', ax=axes[1])
axes[1].set_title('Cap Color Distribution')

plt.tight_layout()
plt.savefig('outputs/fig2_features.png')
plt.close()

# ─────────────────────────────────────────────
# FIGURE 3 – Confusion Matrix
# ─────────────────────────────────────────────
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Edible','Poisonous'],
            yticklabels=['Edible','Poisonous'])

plt.title(f'Confusion Matrix\nAcc={acc:.4f}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/fig3_confusion.png')
plt.close()

# ─────────────────────────────────────────────
# FIGURE 4 – ROC Curve
# ─────────────────────────────────────────────
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0,1],[0,1],'k--')

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/fig4_roc.png')
plt.close()

# ─────────────────────────────────────────────
# FIGURE 5 – Feature Importance
# ─────────────────────────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values().tail(15)

plt.figure(figsize=(8,6))
feat_imp.plot(kind='barh')
plt.title('Top 15 Feature Importance')

plt.tight_layout()
plt.savefig('outputs/fig5_importance.png')
plt.close()

print("\n All figures saved successfully!")