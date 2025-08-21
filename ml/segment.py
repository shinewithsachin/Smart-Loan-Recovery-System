import joblib
from sklearn.cluster import KMeans
import pandas as pd

KMEANS_PATH = "models/kmeans.pkl"

def fit_kmeans(X, k=3):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = km.fit_predict(X)
    joblib.dump(km, KMEANS_PATH)
    return km, clusters

def load_kmeans():
    return joblib.load(KMEANS_PATH)
