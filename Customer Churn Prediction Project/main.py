import pandas as pd
from data_preprocessing import load_data, clean_data
from feature_engineering import add_features
from utils import encode_categoricals, encode_target
from eda import plot_churn_distribution, plot_categorical_vs_churn, cluster_customers, plot_clusters
from modeling import evaluate_models, train_best_model, evaluate_on_holdout
from explainability import explain_with_shap
from sklearn.model_selection import train_test_split

# 1. Load and clean data
df = load_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = clean_data(df)

# 2. Feature engineering
df = add_features(df)

# 3. EDA and customer segmentation (before encoding categoricals!)
plot_churn_distribution(df)
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract']
plot_categorical_vs_churn(df, categorical_cols)
cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumServices']
df, kmeans = cluster_customers(df, cluster_features)
plot_clusters(df, 'tenure', 'MonthlyCharges')

# 4. Encode categoricals and target
df = encode_categoricals(df)
df = encode_target(df)

# 5. Modeling with cross-validation
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
cv_results = evaluate_models(X_train, y_train)
print("Cross-validation ROC AUC scores:")
for model, scores in cv_results.items():
    print(f"{model}: Mean={scores.mean():.3f}, Std={scores.std():.3f}")

# 6. Train best model and evaluate
best_model = train_best_model(X_train, y_train, model_name='XGBoost')
holdout_metrics = evaluate_on_holdout(best_model, X_test, y_test)
print("\nHoldout set metrics:")
for metric, value in holdout_metrics.items():
    print(f"{metric}: {value:.3f}")

# 7. Model interpretation
explain_with_shap(best_model, X_train)

# 8. Business recommendations (to be added in README/markdown)
