# Heart disease datasets

## processed.cleveland.data  (UCI Heart Disease, Cleveland subset)
Source: https://archive.ics.uci.edu/dataset/45/heart+disease  (Detrano et al.)
Documentation: `heart-disease.names` (downloaded alongside).

303 rows, no header, comma separated. 13 features + target:

| column   | type        | meaning (from .names file) |
|----------|-------------|----------------------------|
| age      | numeric     | years |
| sex      | categorical | 1 = male, 0 = female |
| cp       | categorical | chest pain type: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic |
| trestbps | numeric     | resting blood pressure (mm Hg) |
| chol     | numeric     | serum cholesterol (mg/dl) |
| fbs      | categorical | fasting blood sugar > 120 mg/dl (1/0) |
| restecg  | categorical | 0 normal, 1 ST-T abnormality, 2 LV hypertrophy |
| thalach  | numeric     | max heart rate achieved |
| exang    | categorical | exercise induced angina (1/0) |
| oldpeak  | numeric     | ST depression induced by exercise |
| slope    | categorical | 1 upsloping, 2 flat, 3 downsloping |
| ca       | numeric     | number of major vessels coloured by fluoroscopy (0-3); 4 missing as `?` |
| thal     | categorical | 3 normal, 6 fixed defect, 7 reversible defect; 2 missing as `?` |
| num      | target      | angiographic status: 0 = <50% narrowing, 1-4 = >50% narrowing |

**Target used by the platform:** `label = 1 if num > 0 else 0`.
This binarisation is what the dataset's own documentation describes
("distinguish presence (values 1,2,3,4) from absence (value 0)").
Result: 164 absent / 139 present.

## heart_disease_200_synthetic.csv
Local synthetic file of unknown provenance. 200 rows, 4 numeric features
(age, blood_pressure, cholesterol, max_heart_rate) and a `target` column.
**The meaning of `target` is not documented.** The platform treats 1 as
"disease present" purely as a stated assumption; nothing medical should be
inferred from results on this file. Kept as a small smoke-test dataset.
