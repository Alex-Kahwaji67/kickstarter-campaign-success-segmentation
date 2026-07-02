#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Developing the Classification Model ###

# Load Libraries
import pandas as pd
import numpy as np
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore", message="Features .* are constant.", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

# Import data
kickstarter_df = pd.read_csv("data/Kickstarter_data.csv", low_memory=False)

# Filter to successful and failed campaigns only
kickstarter_df = kickstarter_df[kickstarter_df["state"].isin(["successful", "failed"])].copy()

# Target variable: 1 = successful, 0 = failed
y = (kickstarter_df["state"] == "successful").astype(int)

# Engineer pre-launch duration features
kickstarter_df["campaign_duration_days"] = (
    pd.to_datetime(kickstarter_df["deadline"]) - pd.to_datetime(kickstarter_df["launched_at"])
).dt.days
kickstarter_df["prep_time_days"] = (
    pd.to_datetime(kickstarter_df["launched_at"]) - pd.to_datetime(kickstarter_df["created_at"])
).dt.days

# Define predictors - launch-time features only (no post-launch leakage)
features = [
    "goal", "name_len_clean", "blurb_len_clean", "deadline_month", "deadline_hr",
    "launched_at_month", "launched_at_day", "launched_at_hr", "launched_at_yr",
    "created_at_hr", "deadline_weekday", "launched_at_weekday",
    "main_category", "country", "location_type", "disable_communication",
    "show_feature_image", "video", "creator_name_len_clean", "prelaunch_activated",
    "campaign_duration_days", "prep_time_days"
]
X = kickstarter_df[features]

# Drop rows with missing values
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

print(X.shape)
print(y.value_counts())


# In[2]:


# Split into 70% training and 30% test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Define column groups by type
numeric_cols = [
    "goal", "name_len_clean", "blurb_len_clean", "deadline_month", "deadline_hr",
    "launched_at_month", "launched_at_day", "launched_at_hr", "launched_at_yr",
    "created_at_hr", "creator_name_len_clean", "campaign_duration_days", "prep_time_days"
]
categorical_cols = [
    "main_category", "country", "location_type", "deadline_weekday", "launched_at_weekday"
]
binary_cols = [
    "disable_communication", "show_feature_image", "video", "prelaunch_activated"
]

print(f'X_train: {X_train.shape} | X_test: {X_test.shape}')


# In[3]:


# Preprocessor: scale numerics, one-hot encode categoricals, pass through binaries
# Binary columns are already [0,1] - StandardScaler not applied to them
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ("bin", "passthrough", binary_cols)
])


# In[4]:


# 5-fold stratified CV on full preprocessing + feature selection pipelines
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidate_models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42),
}

results = {}
for name, mdl in candidate_models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("selector", SelectKBest(f_classif, k=30)),
        ("model", mdl)
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    results[name] = {"Mean CV Accuracy": scores.mean().round(4), "Std": scores.std().round(4)}

cv_results = pd.DataFrame(results).T.sort_values('Mean CV Accuracy', ascending=False)
print(cv_results)


# In[5]:


# ANN evaluated in the same pipeline framework
ann = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

ann_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("selector", SelectKBest(f_classif, k=30)),
    ("model", ann)
])
ann_scores = cross_val_score(ann_pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
results["ANN"] = {"Mean CV Accuracy": ann_scores.mean().round(4), "Std": ann_scores.std().round(4)}

cv_results = pd.DataFrame(results).T.sort_values('Mean CV Accuracy', ascending=False)
print(cv_results)


# In[6]:


# Select best model by CV accuracy
best_name = cv_results['Mean CV Accuracy'].idxmax()
best_model = ann if best_name == 'ANN' else candidate_models[best_name]

print(f'Best model: {best_name} | CV Accuracy: {cv_results.loc[best_name, "Mean CV Accuracy"]}')


# In[7]:


# Build final pipeline and evaluate on test set
final_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("selector", SelectKBest(f_classif, k=30)),
    ("model", best_model)
])
final_pipe.fit(X_train, y_train)
y_pred = final_pipe.predict(X_test)

print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}\n')
print(classification_report(y_test, y_pred, target_names=['Failed', 'Successful']))

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm,
    index=['Actual: Failed', 'Actual: Successful'],
    columns=['Pred: Failed', 'Pred: Successful'])
print(cm_df)


# # Clustering

# In[9]:


from sklearn.decomposition import PCA

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

# Apply same preprocessor (already fitted on training data)
X_full_processed = preprocessor.transform(X)

# Fit PCA on training data only - transform full dataset for clustering
pca = PCA(n_components=10, random_state=42)
pca.fit(preprocessor.transform(X_train))
X_pca = pca.transform(X_full_processed)

# Variance summary table
evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)
variance_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(10)],
    "Explained Variance": evr.round(4),
    "Cumulative": cum_evr.round(4)
}).set_index('Component')
print(variance_df)


# In[10]:


if plt is not None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(variance_df.index, evr * 100, color='steelblue', edgecolor='black', label='Per Component')
    ax.plot(variance_df.index, cum_evr * 100, color='crimson', marker='o', linewidth=2, label='Cumulative')
    ax.axhline(70, color='gray', linestyle='--', linewidth=1, label='70% threshold')
    ax.set_title('PCA Scree Plot - Explained Variance by Component')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('pca_scree.png', dpi=150)
    plt.show()
else:
    print("matplotlib is not installed; skipping PCA scree plot.")


# In[11]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method: k = 2 to 10
inertias = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    inertias[k] = km.inertia_

# Silhouette score with sample_size to manage memory
sil_scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labs = km.fit_predict(X_pca)
    sil_scores[k] = round(silhouette_score(X_pca, labs, sample_size=3000, random_state=42), 4)

print('Silhouette Scores:')
print(pd.Series(sil_scores, name='Score').rename_axis('k'))


# In[12]:


if plt is not None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(list(inertias.keys()), list(inertias.values()), marker='o', color='steelblue', linewidth=2)
    axes[0].set_title('Elbow Method - KMeans Inertia')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_xticks(range(2, 11))

    axes[1].plot(list(sil_scores.keys()), list(sil_scores.values()), marker='o', color='crimson', linewidth=2)
    axes[1].set_title('Silhouette Score vs k')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_xticks(range(2, 8))

    plt.tight_layout()
    plt.savefig('kmeans_selection.png', dpi=150)
    plt.show()
else:
    print("matplotlib is not installed; skipping KMeans selection plot.")


# In[13]:


from sklearn.cluster import DBSCAN

# Test DBSCAN across epsilon configurations
dbscan_results = []
for eps in [0.5, 1.0, 1.5]:
    labs = DBSCAN(eps=eps, min_samples=5).fit_predict(X_pca)
    noise = (labs == -1).sum()
    n_clusters = len(set(labs[labs != -1]))
    dbscan_results.append({
        'Algorithm': 'DBSCAN', 'Config': f'eps={eps}',
        'Clusters': n_clusters, 'Noise Points': noise,
        'Noise %': round(noise / len(labs) * 100, 1)
    })

print(pd.DataFrame(dbscan_results).to_string(index=False))


# In[16]:


# Final model: k=3 selected by silhouette score (tied k=3 and k=4; k=3 preferred for parsimony)
best_k = 3
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_pca)

# Build profile using original feature matrix
cluster_df = X.copy().reset_index(drop=True)
cluster_df['cluster'] = cluster_labels
cluster_df['state'] = y.values

profile_cols = ['goal', 'campaign_duration_days', 'prep_time_days', 'name_len_clean', 'blurb_len_clean']
profile = cluster_df.groupby('cluster')[profile_cols].mean().round(2)
profile['success_rate'] = cluster_df.groupby('cluster')['state'].mean().round(4)
profile['n'] = cluster_df.groupby('cluster').size()

print(f'Final k: {best_k}')
print(profile)

# Top category per cluster
for c in range(best_k):
    top = cluster_df[cluster_df['cluster']==c]['main_category'].value_counts().head(4)
    print(f'Cluster {c} top categories: {top.to_dict()}')


# In[ ]:





