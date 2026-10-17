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

# Step 5 - add_ratio_features (not yet solved)
# TODO: implement

# Step 6 - split_features_labels (not yet solved)
# TODO: implement

# Step 7 - ClusterSimilarity (not yet solved)
# TODO: implement

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

