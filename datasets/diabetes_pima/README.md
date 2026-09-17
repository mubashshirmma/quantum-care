# Pima Indians Diabetes dataset

Source: National Institute of Diabetes and Digestive and Kidney Diseases; distributed via UCI
(now removed there) and mirrored at https://github.com/jbrownlee/Datasets (pima-indians-diabetes.data.csv).
Population: female patients of Pima Indian heritage, age >= 21. 768 rows, 8 features.

| column | meaning |
|---|---|
| pregnancies | number of times pregnant |
| glucose | plasma glucose concentration, 2 h in an oral glucose tolerance test (mg/dl) |
| blood_pressure | diastolic blood pressure (mm Hg) |
| skin_thickness | triceps skin fold thickness (mm) |
| insulin | 2-hour serum insulin (mu U/ml) |
| bmi | body mass index (kg/m^2) |
| diabetes_pedigree | diabetes pedigree function |
| age | years |
| outcome | **1 = tested positive for diabetes** (WHO criteria), 0 = negative |

Known data issue: zeros in glucose, blood_pressure, skin_thickness, insulin and bmi are
physiologically impossible and encode *missing values*. The platform treats those zeros as
missing (`zero_as_missing`) and imputes them from the training split.
Class balance: 500 negative / 268 positive.
