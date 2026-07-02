# Kickstarter Campaign Success And Segmentation

This project predicts whether Kickstarter campaigns will succeed using launch-time features, then segments campaigns into creator archetypes with PCA and clustering.

The work began as an individual Data Mining for Business Analytics project and has been staged here for portfolio review.

## Problem

Kickstarter campaigns follow an all-or-nothing funding model. Before a campaign raises money, creators and the platform need signals about whether the campaign is likely to succeed and what kind of launch profile it resembles.

This project answers two questions:

- Can campaign success be predicted using only information available at launch?
- Do campaigns cluster into interpretable creator archetypes based on ambition, timing, and preparation?

## Data

The submitted project used 25,792 Kickstarter campaigns from 2018 to 2024 after filtering to successful and failed campaigns and dropping rows with missing selected launch-time fields.

Raw data is not included in this staged repo. To run the notebook, place the source files here:

```text
data/Kickstarter_data.csv
data/Kickstarter-Grading.csv
```

The grading file was part of the course evaluation workflow and is optional for the public portfolio version.

## Methods

- Leakage control by excluding post-launch variables such as pledged amount, backer count, spotlight status, USD pledged, and state-change timestamps.
- Feature engineering for campaign duration and creator preparation time.
- Preprocessing with scaled numeric features, one-hot encoded categorical features, and passthrough binary flags.
- Model comparison with 5-fold stratified cross-validation.
- Classification models: logistic regression, KNN, decision tree, random forest, and artificial neural network.
- Feature selection with `SelectKBest` inside the modeling pipeline.
- PCA to reduce the encoded feature space before clustering.
- KMeans and DBSCAN comparison for campaign segmentation.

## Results

The artificial neural network was selected as the strongest classifier:

- Mean cross-validation accuracy: 0.7240
- Held-out test accuracy: 0.7190
- Macro F1: 0.68
- Successful-campaign recall: 0.85
- Failed-campaign recall: 0.49

For clustering, 10 PCA components explained 70.52% of variance. KMeans with `k=3` produced the most interpretable segmentation:

- High-investment, well-prepared campaigns: highest goals, longest preparation time, highest success rate.
- Mid-scale underperformers: moderate goals and preparation, lowest success rate.
- Lean, quick-launch campaigns: lowest goals, shortest preparation time, near-average success rate.

## How To Run

1. Use Python 3.10+.
2. Install the key libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

3. Place the dataset files in `data/`.
4. Open and run:

```text
notebooks/kickstarter_campaign_success_segmentation.ipynb
```

## Portfolio Notes

This project is promising, but it should be lightly cleaned before publishing:

- Confirm the dataset source/license before linking or distributing data.
- Add a short model-limitations section explaining the weaker recall for failed campaigns.
- Consider adding a license once the intended public repo policy is clear.
