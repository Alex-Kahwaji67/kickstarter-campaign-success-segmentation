# Kickstarter Campaign Success and Segmentation

This project analyzes whether Kickstarter campaign outcomes can be predicted from information available at launch. I built a supervised classification model to predict campaign success, then used PCA and clustering to identify campaign archetypes based on goal size, preparation time, timing, and category structure.

The main focus was avoiding leakage. Fields such as pledged amount, backer count, spotlight status, USD pledged, and state-change timestamps were excluded because they would not be available before a campaign starts receiving funding.

## Business Question

Kickstarter campaigns follow an all-or-nothing funding model. Before a campaign launches, creators and the platform could benefit from an early signal of likely success.

This project asks:

- Can launch-time campaign features predict whether a campaign will succeed?
- Can campaigns be grouped into interpretable creator archetypes?
- What practical recommendations could Kickstarter make before launch?

## Data

The analysis used 25,792 Kickstarter campaigns from 2018 to 2024 after filtering to campaigns marked either successful or failed and removing rows with missing selected launch-time fields.

Raw data is not included in this repository. To reproduce the analysis, place the dataset at:

```text
data/Kickstarter_data.csv
```

## Approach

The classification task used 22 launch-time predictors, including goal amount, campaign timing, category, country, media flags, prelaunch activation, campaign duration, and creator preparation time.

The modeling workflow included:

- Train/test split with stratification
- Standard scaling for numeric features
- One-hot encoding for categorical features
- Passthrough handling for binary fields
- Feature selection with `SelectKBest`
- 5-fold stratified cross-validation
- Model comparison across logistic regression, KNN, decision tree, random forest, and neural network classifiers

For segmentation, I applied PCA to the processed feature space and compared KMeans against DBSCAN.

## Results

The neural network performed best in cross-validation and was selected as the final classifier.

| Metric | Result |
|---|---:|
| Mean CV accuracy | 0.7240 |
| Test accuracy | 0.7190 |
| Macro F1 | 0.68 |
| Successful campaign recall | 0.85 |
| Failed campaign recall | 0.49 |

The model was stronger at identifying successful campaigns than failed campaigns. That matters because a platform intervention tool would likely care most about identifying at-risk campaigns before launch.

For clustering, 10 PCA components explained 70.52% of total variance. KMeans with `k=3` produced the clearest segmentation:

| Segment | Description |
|---|---|
| High-investment, well-prepared campaigns | Higher goals, longer preparation time, highest success rate |
| Mid-scale underperformers | Moderate goals and preparation time, lowest success rate |
| Lean, quick-launch campaigns | Lower goals, shorter preparation time, near-average success rate |

## Takeaways

The strongest insight was not just the classifier accuracy, but the relationship between ambition and preparation. Campaigns with higher funding goals were not necessarily weaker when they also showed longer preparation time.

For a platform like Kickstarter, this suggests three possible product interventions:

- Give creators a launch-readiness score before publishing
- Flag mid-scale campaigns that may need goal calibration
- Recommend longer preparation windows for high-goal campaigns

## Limitations

The model has weaker recall for failed campaigns, meaning it misses some campaigns that eventually fail. Future work could improve this by optimizing for recall or F1 instead of accuracy, testing class-weighting and threshold tuning, and adding richer text features from campaign descriptions.

The dataset source and redistribution rights should be confirmed before publishing raw data. For that reason, the CSV is intentionally excluded from this repository.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
data/Kickstarter_data.csv
```

Run the script:

```bash
python src/kickstarter_individual_project.py
```

Or open the notebook:

```text
notebooks/kickstarter_campaign_success_segmentation.ipynb
```
