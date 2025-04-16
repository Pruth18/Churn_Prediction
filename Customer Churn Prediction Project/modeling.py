import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, tree_method='hist', gpu_id=-1)
    }


def evaluate_models(X, y, cv=5):
    models = get_models()
    results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')
        results[name] = scores
    return results


def train_best_model(X, y, model_name='XGBoost'):
    models = get_models()
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', models[model_name])
    ])
    pipeline.fit(X, y)
    return pipeline


def evaluate_on_holdout(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
