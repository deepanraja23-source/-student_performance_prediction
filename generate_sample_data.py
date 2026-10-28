"""Creates a SYNTHETIC sample dataset (data/students.csv) so the project runs out of the box.
Replace data/students.csv with your real dataset (keep the column names or edit config in student_prediction.py)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 500
attendance = np.clip(rng.normal(78, 12, n), 35, 100)
study_hours = np.clip(rng.normal(3, 1.3, n), 0, 8)
previous_marks = np.clip(rng.normal(65, 14, n), 20, 100)
assignment_marks = np.clip(rng.normal(70, 12, n), 20, 100)
internal_marks = np.clip(0.4*previous_marks + 0.3*assignment_marks + 4*study_hours + rng.normal(0, 6, n), 10, 100)
score = (0.25*attendance + 4*study_hours + 0.3*previous_marks + 0.2*internal_marks + rng.normal(0, 5, n))
result = np.where(score >= np.median(score) - 3, "Pass", "Fail")
df = pd.DataFrame({"attendance": attendance.round(1), "study_hours": study_hours.round(1),
                   "previous_marks": previous_marks.round(1), "assignment_marks": assignment_marks.round(1),
                   "internal_marks": internal_marks.round(1), "result": result})
# add a few missing values to demonstrate cleaning
for col in ["attendance", "study_hours"]:
    df.loc[rng.choice(n, 8, replace=False), col] = np.nan
df.to_csv("data/students.csv", index=False)
print("Saved data/students.csv", df.shape)
