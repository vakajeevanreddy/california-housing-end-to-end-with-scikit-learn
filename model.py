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

# Step 8 - numeric_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
def numeric_pipeline():
    # TODO: make_pipeline(SimpleImputer(median), StandardScaler())
    return make_pipeline(
        SimpleImputer(strategy= "median"),
        StandardScaler()
    )

# Step 9 - categorical_pipeline
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def categorical_pipeline():
    return make_pipeline(
        SimpleImputer(missing_values = None,strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

# Step 10 - build_preprocessing
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.compose import ColumnTransformer
def build_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    # TODO: ColumnTransformer with 'log', 'geo', 'cat' transformers and remainder=numeric_pipeline().
    log_pipeline = make_pipeline(
        SimpleImputer(strategy = "median"),
        FunctionTransformer(np.log, feature_names_out = "one-to-one"),
        StandardScaler()
    )
    geo_transformer = ClusterSimilarity(
        n_clusters = n_clusters,
        gamma = gamma,
        random_state = random_state
    )
   
    preprocess = ColumnTransformer(
        transformers = [
        ("log" , log_pipeline,
        ["total_bedrooms","total_rooms","population", "households" , "median_income"]),
        ("geo", geo_transformer,
        ["latitude","longitude"]),
        ("cat" , categorical_pipeline() , ["ocean_proximity"])
        ],
        remainder = numeric_pipeline()
    )
    return preprocess

# Step 11 - rmse
import numpy as np
import pandas as pd
def rmse(y_true, y_pred):
    # TODO: sqrt(mean((y_true - y_pred)^2)) as a float.
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be match")
    return float(np.sqrt(np.mean((y_true-y_pred)** 2 )))

# Step 12 - dummy_baseline_rmse
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
def dummy_baseline_rmse(X, y):
    # TODO: fit DummyRegressor(strategy='mean') and return its RMSE on (X, y).
    dummy = DummyRegressor(strategy = "mean")
    dummy.fit(X,y)
    y_pred = dummy.predict(X)
    return rmse(y,y_pred)

# Step 13 - cross_val_rmse
import numpy as np
from sklearn.model_selection import cross_val_score
def cross_val_rmse(model, X, y, cv=3):
    # TODO: cross_val_score with neg_root_mean_squared_error; return {'scores': [...], 'mean': ..., 'std': ...}.
    scores = cross_val_score(model,X,y,
                cv = cv,scoring = "neg_root_mean_squared_error")
    rmse_scores = -scores
    return {
        "scores" : rmse_scores.tolist(),
        "mean" : float(np.mean(rmse_scores)),
        "std" : float(np.std(rmse_scores))
    }

# Step 14 - linear_model
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
def linear_model(preprocessing):
    # TODO: make_pipeline(preprocessing, LinearRegression())
    return make_pipeline(preprocessing, LinearRegression())

# Step 15 - forest_model
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
def forest_model(preprocessing, n_estimators=50, random_state=42):
    # TODO: make_pipeline(preprocessing, RandomForestRegressor(n_estimators, random_state))
    return make_pipeline(preprocessing, RandomForestRegressor(
        n_estimators = n_estimators, random_state = random_state
    ))

# Step 16 - random_search
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

def random_search(pipeline, X, y, n_iter=5, cv=3, random_state=42):
    param_distributions = {
        "columntransformer__geo__n_clusters": list(range(3, 11)),
        "randomforestregressor__max_features": list(range(2, 9))
    }
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X, y)
    return search

# Step 17 - test_rmse
import numpy as np
def test_rmse(model, test_set):
    # TODO: add ratio features, split, predict with the fitted model, return rmse.
    test_set = add_ratio_features(test_set)
    X_test = test_set.drop("median_house_value", axis = 1)
    y_test = test_set["median_house_value"]
    predictions = model.predict(X_test)
    return rmse(y_test,predictions)

# Step 18 - bootstrap_rmse_ci
def bootstrap_rmse_ci(y_true, y_pred, n_boot=200, alpha=0.05, random_state=42):
    # TODO: bootstrap the RMSE by resampling index pairs; return (low, high) percentiles as floats.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)
    boot_rmses = []
    for _ in range(n_boot):
        indicies = rng.integers(0, n , n)
        y_resampled = y_true[indicies]
        yhat_resampled = y_pred[indicies]
        boot_rmses.append(rmse(y_resampled,yhat_resampled))
    low = float(np.percentile(boot_rmses , 100 * alpha / 2))
    high = float(np.percentile(boot_rmses, 100 * (1 - alpha/2)))
    return low,high

# Step 19 - feature_importances
import numpy as np
def feature_importances(search, k=5):
    # TODO: top-k (importance, name) tuples from the best estimator, importances rounded to 3 decimals.
    pipeline = search.best_estimator_
    feature_names = pipeline.steps[0][1].get_feature_names_out()
    model = pipeline.steps[-1][1]
    importances = model.feature_importances_
    pairs =[(float(round(imp,3)),name)for imp,name in zip(importances, feature_names)]
    pairs_sorted = sorted(pairs,key = lambda x : x[0], reverse = True)
    return pairs_sorted[:k]

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

