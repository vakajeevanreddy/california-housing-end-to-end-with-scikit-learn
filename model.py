"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
def load_housing():
    # TODO: Download housing.tgz once into tempfile.gettempdir() and read housing/housing.csv from it.
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    tgz_path = os.path.join(tempfile.gettempdir(), "housing.tgz")
    if not os.path.exists(tgz_path):
        urllib.request.urlretrieve(url,tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path = tempfile.gettempdir())
    csv_path = os.path.join(tempfile.gettempdir(), "housing" , "housing.csv")
    housing_df = pd.read_csv(csv_path)
    return housing_df

# Step 2 - income_categories
import numpy as np
import pandas as pd

def income_categories(df):
    # TODO: pd.cut median_income with edges [0, 1.5, 3, 4.5, 6, inf] and labels 1..5; return an int Series.
    bins = [0, 1.5, 3.0, 4.5, 6.0, np.inf]
    labels = [1, 2, 3, 4, 5]
    categories = pd.cut(df["median_income"], bins = bins, labels = labels)
    return categories.astype(int)

# Step 3 - stratified_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(df, test_size=0.2, random_state:int = 42):
    # TODO: train_test_split stratified on income_categories(df); return (train_set, test_set).
    start_labels = income_categories(df)
    train_set,test_set = train_test_split(df,test_size = test_size,random_state = random_state,stratify = start_labels)
    return train_set,test_set

# Step 4 - explore_correlations
def explore_correlations(df):
    # TODO: Pearson correlation of every numeric column with median_house_value, sorted descending, target excluded.
    corr_matrix = df.corr(numeric_only = True)
    target_corr = corr_matrix["median_house_value"].drop("median_house_value")
    return target_corr.sort_values(ascending = False)

# Step 5 - add_ratio_features
def add_ratio_features(df):
    # TODO: Return a copy with rooms_per_house, bedrooms_ratio and people_per_house columns added.
    new_df = df.copy()
    new_df["rooms_per_house"] = new_df["total_rooms"] / new_df["households"]
    new_df["bedrooms_ratio"] = new_df["total_bedrooms"]/new_df["total_rooms"]
    new_df["people_per_house"] = new_df["population"] / new_df["households"]
    return new_df

# Step 6 - split_features_labels
def split_features_labels(df):
    # TODO: Return (X without median_house_value, y = median_house_value Series).
    x = df.drop("median_house_value", axis = 1)
    y = df["median_house_value"].copy()
    return x,y

# Step 7 - ClusterSimilarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        # Store parameters
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        # Fit KMeans clustering
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        # RBF kernel similarity of each row of X to cluster centers
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        # Feature names for similarity features
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Step 8 - numeric_pipeline (not yet solved)
# TODO: implement

# Step 9 - categorical_pipeline (not yet solved)
# TODO: implement

# Step 10 - build_preprocessing (not yet solved)
# TODO: implement

# Step 11 - rmse (not yet solved)
# TODO: implement

# Step 12 - dummy_baseline_rmse (not yet solved)
# TODO: implement

# Step 13 - cross_val_rmse (not yet solved)
# TODO: implement

# Step 14 - linear_model (not yet solved)
# TODO: implement

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

