import pandas as pd, numpy as np, glob
names=["LogReg","GaussianNB","KNN","RandomForest","LightGBM","MLP","LinearSVM"]
cv=pd.concat([pd.read_csv(f"../results/cv_folds_{n}.csv") for n in names]); cv.to_csv("../results/cv_folds_all_models.csv",index=False)
st=pd.concat([pd.read_csv(f"../results/student_results_{n}.csv") for n in names])
g=cv.groupby("model").agg(cv_macro_f1=("macro_f1","mean"),cv_macro_f1_std=("macro_f1","std"),cv_top1=("top1","mean"),cv_top1_std=("top1","std"),cv_log_loss=("log_loss","mean")).reset_index()
t=st.merge(g,on="model")
t=t[["model","family","config","cv_macro_f1","cv_macro_f1_std","cv_top1","cv_top1_std","cv_log_loss","st_top1","st_macro_f1","st_accuracy","st_precision","st_recall","st_log_loss","st_cont_strict","st_cont_relaxed","st_lcs_f1","st_len_f1","train_time_s","infer_time_ms_per_obs"]]
t=t.set_index("model").loc[names].reset_index(); t.to_csv("../results/model_comparison.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print(t.drop(columns=["config","family"]).round(4).to_string(index=False))
