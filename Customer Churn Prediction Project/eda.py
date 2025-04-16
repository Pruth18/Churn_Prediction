import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def plot_churn_distribution(df: pd.DataFrame):
    plt.figure(figsize=(6,4))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.show()


def plot_categorical_vs_churn(df: pd.DataFrame, columns):
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(columns, 1):
        plt.subplot(2, 4, i)
        sns.countplot(x=column, hue='Churn', data=df)
        plt.title(f'{column} vs Churn')
        plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def cluster_customers(df: pd.DataFrame, features, n_clusters=4):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    return df, kmeans


def plot_clusters(df: pd.DataFrame, x_col, y_col):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=x_col, y=y_col, hue='Cluster', palette='Set1', data=df, alpha=0.7)
    plt.title(f'Customer Segments: {x_col} vs {y_col}')
    plt.show()
