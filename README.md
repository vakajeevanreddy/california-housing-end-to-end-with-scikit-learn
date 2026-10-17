# California Housing, End to End with Scikit-Learn

The classic end-to-end regression project from chapter 2 of Hands-On Machine Learning, done the way it is in practice: download the California housing data, split it honestly, build a leak-proof pipeline with a custom cluster-similarity transformer, tune with RandomizedSearchCV, report test RMSE with a bootstrap interval, then save and serve it.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_housing
- [ ] **2.** income_categories
- [ ] **3.** stratified_split
- [ ] **4.** explore_correlations
- [ ] **5.** add_ratio_features
- [ ] **6.** split_features_labels
- [ ] **7.** ClusterSimilarity
- [ ] **8.** numeric_pipeline
- [ ] **9.** categorical_pipeline
- [ ] **10.** build_preprocessing
- [ ] **11.** rmse
- [ ] **12.** dummy_baseline_rmse
- [ ] **13.** cross_val_rmse
- [ ] **14.** linear_model
- [ ] **15.** forest_model
- [ ] **16.** random_search
- [ ] **17.** test_rmse
- [ ] **18.** bootstrap_rmse_ci
- [ ] **19.** feature_importances
- [ ] **20.** worst_errors
- [ ] **21.** save_and_reload
- [ ] **22.** predict_new

---

Built on Deep-ML.
