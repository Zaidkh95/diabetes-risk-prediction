# Predicting Diabetes Risk from a Health Survey

An end-to-end **classical machine learning** project (no deep learning) that predicts whether a person is **at risk of diabetes or prediabetes** using only answers to a free health questionnaire. The goal is a low-cost screening tool that helps decide *who should get a blood test first*.

**Dataset:** CDC BRFSS 2015, *Diabetes Health Indicators* (Kaggle, by Alex Teboul). 253,680 survey responses, 21 features.
**Final model:** Tuned HistGradientBoosting (scikit-learn)
**Test ROC-AUC:** 0.8145 · **Test recall (at-risk caught):** 79.2% · **Test precision:** 34.3%

---

## 1. Problem framing

- **Type:** Binary classification. The 3-class target (healthy / prediabetes / diabetes) was merged into *at risk (1)* = prediabetes or diabetes, vs *healthy (0)*. Prediabetes is only ~2% of rows, and both groups lead to the same next step: a blood test.
- **Primary metric:** ROC-AUC. It is threshold-independent and robust to the 83/17 class imbalance.
- **Secondary metrics:** Recall (a missed case is the costliest error in screening), PR-AUC, F1.
- **Why not accuracy:** A model that answers "healthy" for everyone scores about 83% accuracy and catches 0% of at-risk people.

## 2. Data

| Item | Value |
|---|---|
| Rows (after removing 23,899 exact duplicates) | 229,781 |
| Features | 21 survey answers (blood pressure, cholesterol, BMI, smoking, activity, diet, general health, healthcare access, age, education, income) |
| Target | `Diabetes_012`: 0 = no diabetes, 1 = prediabetes, 2 = diabetes |
| Missing values | None |
| Split | Stratified 70 / 15 / 15 (train / validation / test), `random_state=42` |

Duplicates were removed **before** splitting so identical rows cannot appear in both train and test.

## 3. Key EDA findings (training set only)

- **High blood pressure** raises the at-risk rate about 3.4× (from 8% to 28%).
- Risk climbs with **age**, **BMI** and **poor self-rated health**, and falls with **income** and **education**.
- Physical activity is associated with lower risk (24% → 15%).
- BMI has a long right tail (max 98, an impossible value). It is **winsorised** at the training 0.5th and 99.5th percentiles.

## 4. Feature engineering

All new features are row-wise formulas, so they cannot leak information.

| Feature | Formula | Why |
|---|---|---|
| `Metabolic_Risk` | BMI × (1 + HighBP + HighChol) | Excess weight is more dangerous alongside high blood pressure and cholesterol. The "1 +" stops obese people with no diagnosis from scoring 0. |
| `Age_x_BMI` | Age × BMI | Obesity is more dangerous at older ages, an interaction linear models cannot learn alone. |
| `Frailty_Index` | GenHlth × (1 + DiffWalk) + PhysHlth / 30 | Physical frailty: poor health weighs more when mobility is limited. |
| `CardioRiskCount` | HighBP + HighChol + HeartDisease + Stroke | Cardiovascular risk factors compound. |
| `Sedentary_BMI`, `Socioeconomic_Score`, `BMI_Category`, `MentHlth_log`, `PhysHlth_log` | Various | Activity-adjusted BMI, income × education, WHO BMI classes, and log transforms for zero-inflated counts. |

**Feature selection** used a drop test with 3-fold CV on the training set, not correlation alone. Five weak features (`Fruits`, `Veggies`, `Smoker`, `AnyHealthcare`, `NoDocbcCost`) cost only about 0.0003 AUC when removed, so they were dropped. The "weak-looking" group `Sex`, `HvyAlcoholConsump`, `CholCheck` cost about 10× more, so it was kept.

## 5. Models compared (validation set)

Five models plus a dummy baseline, all with imbalance handling:

| Model | Imbalance handling |
|---|---|
| Logistic Regression | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced_subsample"` |
| HistGradientBoosting | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight` |
| LightGBM | `class_weight="balanced"` |

Dummy baseline: ROC-AUC 0.5, recall 0. Gradient boosting models led the comparison, and HistGradientBoosting was selected as the best model.

## 6. Tuning and threshold

- **RandomizedSearchCV** (25 candidates, 3-fold stratified CV on training data only). CV ROC-AUC went from 0.8082 (default) to 0.8085 (tuned). The gain is small, which suggests the survey features are the limiting factor, not model capacity.
- **Decision threshold** tuned on validation. The screening threshold of **0.487** targets at least 80% recall. The max-F1 alternative is 0.621.

## 7. Final test results

Tuned model, threshold 0.487, evaluated **once** on the held-out test set (34,468 people):

| Metric | Test |
|---|---|
| ROC-AUC | **0.8145** |
| PR-AUC (no-skill = 0.173) | 0.4727 |
| Recall | 0.7924 (catches 79% of at-risk people) |
| Precision | 0.3425 |
| F1 | 0.4782 |
| Accuracy | 0.7011 |

**Top drivers (permutation importance):** `Metabolic_Risk`, `Age_x_BMI`, `GenHlth`, `Frailty_Index`, `CholCheck`. Three of the top four are engineered features.

## 8. Error analysis

- About **21%** of at-risk people in the test set are missed (1,237 of 5,959).
- Missed people look healthy on paper: younger, lower BMI, and most report no high blood pressure or cholesterol.
- Miss rates are highest for at-risk people aged 18–39 (about 63%) and those with zero cardiovascular risk factors (about 75%).
- Roughly 4 in 10 missed people score within 0.10 of the threshold. About 1 in 4 score below 0.25, so this survey cannot see them.

## 9. Limitations

- **Self-reported data.** Recall and social-desirability bias affect answers, and people with undiagnosed diabetes are labelled 0.
- **No lab values or key risk factors.** Glucose, HbA1c, family history, gestational diabetes, ethnicity and waist circumference are not in the survey.
- **Cross-sectional, U.S., 2015.** These are associations, not causes, and the model needs re-validation for other populations or years.
- **Modest precision.** The model is a triage tool, not a diagnosis.
- **Calibration.** Class weighting inflates predicted probabilities, so recalibrate before reporting individual risk percentages.

## 10. Next steps

1. Probability calibration (`CalibratedClassifierCV`, isotonic).
2. Fairness audit: compare recall across sex, age and income groups.
3. Try CatBoost and a stacking ensemble.
4. SHAP explanations for individual predictions.
5. Validate on newer BRFSS years (2019–2023) to check for drift.

---

## Repository contents

```
├── Diabetes_ML_Project_Final_Project_Zaid_Salem_Alkharabsheh.ipynb   # full analysis
├── Diabetes_Risk_Prediction_-_Zaid_Salem_Alkharabsheh_QAFZA.pdf      # presentation
├── requirements.txt
└── README.md
```

The dataset is not included. Download `diabetes_012_health_indicators_BRFSS2015.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data).

## How to run

```bash
pip install -r requirements.txt
# place the CSV in the same folder as the notebook, or set DIABETES_CSV=/path/to/file.csv
jupyter notebook Diabetes_ML_Project_Final_Project_Zaid_Salem_Alkharabsheh.ipynb
```

The notebook uses `RANDOM_STATE = 42` throughout, so results are reproducible. Random search takes about 5 minutes.

## Author

**Zaid Salem Alkharabsheh** · QAFZA Tech · November 2026
