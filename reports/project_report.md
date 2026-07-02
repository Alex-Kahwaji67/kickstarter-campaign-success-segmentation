# Kickstarter Campaign Success And Segmentation

## Summary

This project used 25,792 Kickstarter campaigns from 2018 to 2024 to predict campaign success at launch and segment campaigns into creator archetypes. The analysis used only launch-time predictors and excluded post-launch fields such as pledged amount, backer count, spotlight status, USD pledged, and state-change timestamps to avoid leakage.

## Classification

The model used 22 launch-time predictors covering funding goal, campaign text length, timing, category, country, location type, media flags, prelaunch activation, campaign duration, and creator preparation time.

Preprocessing used:

- Standard scaling for numeric features.
- One-hot encoding for categorical features.
- Passthrough for binary flags.
- `SelectKBest` feature selection inside the modeling pipeline.
- 5-fold stratified cross-validation.

Five classifiers were compared:

| Model | Mean CV Accuracy | Std |
|---|---:|---:|
| ANN | 0.7240 | 0.0075 |
| Random Forest | 0.7113 | 0.0184 |
| Logistic Regression | 0.6919 | 0.0050 |
| Decision Tree | 0.6867 | 0.0124 |
| KNN | 0.6800 | 0.0081 |

The ANN was selected as the final classifier. On the held-out test set it achieved:

- Accuracy: 0.7190
- Macro F1: 0.68
- Weighted F1: 0.71
- Failed-campaign precision/recall/F1: 0.67 / 0.49 / 0.57
- Successful-campaign precision/recall/F1: 0.74 / 0.85 / 0.79

The model is better at identifying successful campaigns than failed campaigns. That is an important limitation for product use: a platform intervention workflow would need threshold tuning or a failure-focused objective if the goal is to help risky campaigns before launch.

## Clustering

The clustering workflow used the same preprocessed feature representation, reduced with PCA. Ten principal components explained 70.52% of variance.

KMeans and DBSCAN were compared. DBSCAN did not produce a stable, interpretable partition across tested epsilon values. KMeans with `k=3` was selected because it gave the clearest business-readable segmentation.

The three clusters were:

| Cluster | Interpretation | Size | Success Rate |
|---|---|---:|---:|
| 0 | High-investment, well-prepared campaigns | 9,082 | 0.6626 |
| 1 | Mid-scale underperformers | 6,530 | 0.5922 |
| 2 | Lean, quick-launch campaigns | 10,180 | 0.6159 |

The strongest business insight is that campaign outcomes appear related to combinations of ambition, preparation, and timing. High-goal campaigns can still outperform when they show longer preparation time, while mid-scale campaigns may need either better preparation or better goal calibration.

## Recommended Portfolio Framing

Position this as a leakage-aware classification and segmentation project, not as a production-ready model. The best interview angle is the analytical judgment: excluding post-launch leakage, comparing model families, acknowledging failed-campaign recall, and translating clusters into product interventions.
