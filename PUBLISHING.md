# Publishing Checklist

Use this checklist before creating the public GitHub repo.

## Privacy

- Do not publish the original submitted PDF because it includes student-identifying information.
- Do not commit `data/Kickstarter_data.csv` unless the dataset source and license are confirmed.
- Keep `.gitignore` in place. It excludes `data/*.csv`.

## Recommended GitHub Repo

- Repository name: `kickstarter-campaign-success-segmentation`
- Description: `Leakage-controlled Kickstarter campaign success classifier with PCA/KMeans creator segmentation.`
- Visibility: Public
- Add README on GitHub: No, use the local README already prepared.

## Files To Publish

Publish:

- `README.md`
- `requirements.txt`
- `.gitignore`
- `PUBLISHING.md`
- `data/README_DATA.md`
- `notebooks/kickstarter_campaign_success_segmentation.ipynb`
- `reports/project_report_scrubbed.md`
- `src/kickstarter_individual_project.py`

Do not publish:

- `data/Kickstarter_data.csv`
- Original submitted PDF
- Original course grading files
- `__pycache__/`
- `.ipynb_checkpoints/`

## Local Verification

From this repo folder:

```bash
pip install -r requirements.txt
python src/kickstarter_individual_project.py
```

Expected key outputs:

- Dataset after filtering/cleaning: `(25792, 22)`
- Selected classifier: ANN
- Mean CV accuracy: `0.7240`
- Test accuracy: `0.7190`
- Macro F1: `0.68`
- Final KMeans cluster count: `3`

## Publish Commands

After creating an empty public GitHub repo:

```bash
git init
git add .
git status
git commit -m "Stage Kickstarter success and segmentation portfolio project"
git branch -M main
git remote add origin https://github.com/Alex-Kahwaji67/kickstarter-campaign-success-segmentation.git
git push -u origin main
```

Before committing, confirm `git status` does not show `data/Kickstarter_data.csv`.
