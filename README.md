# Student Performance Prediction

Summer Internship project - Fantasy Solution, Trichy (01/06/2026 - 30/06/2026)
Deepan Raja S (810023104047), CSE, University College of Engineering (BIT Campus)

## Setup
    pip install -r requirements.txt

## Run
    python generate_sample_data.py     # only if you want the SYNTHETIC sample data
    python student_prediction.py train
    python student_prediction.py predict --attendance 80 --study_hours 3 --previous_marks 65 --assignment_marks 70 --internal_marks 60

## Notes
- `data/students.csv` is synthetic demo data. Replace it with your real dataset and, if needed,
  edit the CONFIG section at the top of `student_prediction.py` (file name, column names, target).
- Outputs (charts, model_comparison.csv, best_model.joblib) are saved in `outputs/`.
- Models compared: Logistic Regression, Decision Tree, Random Forest (80/20 split, 5-fold CV).
