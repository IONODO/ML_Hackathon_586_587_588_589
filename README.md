# PathSnap 

Python 3.12. Install:
    pip install numpy pandas scipy scikit-learn lightgbm optuna shapely pyproj networkx folium fastapi uvicorn matplotlib

## Data placement
Unzip Student_folder.zip so you have:
    data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}      (80 trips)
    data/student_test/<trip_id>/...                              (10 trips)
(i.e. copy "Student folder/train" -> data/train and "Student folder/student_test" -> data/student_test)
The evaluator partition is never loaded by any dev script.

## Run order (cd src first; all paths are relative to src/)
 1. python eda.py                  -> results/eda_inventory.csv  (structural EDA, prints partition summary)
 2. python build_dataset.py        -> results/cand_features.pkl, obs_labels.pkl (alignment labels, candidates R=80,K=25, features) ~35 s
 3. python sweep_candidates.py     -> results/candidate_sweep_train.csv (miss rate vs R,K)   [optional]
 4. python run_comparison.py       -> 7 models, 5-fold group CV + student-test; per-model CSVs in results/
 5. python aggregate_models.py     -> results/model_comparison.csv
 6. python tune_bayes.py 30        -> Optuna TPE on RandomForest; results/bayes_trials.csv, bayes_best.json
 7. python final_tuned.py          -> tuned vs untuned; saves artifacts/rf_tuned.joblib   (NOT YET RUN)
 8. python overfit_check.py        -> train-vs-CV gap + MLP epoch curves                  (NOT YET RUN)
hmm.py (Viterbi decoder) and viz.py (Folium) are written but NOT yet tested/run.
Not started: HMM tuning/eval script, FastAPI, UI, report.
Machine used: 1 CPU, 3 GB RAM. Seed 42.
