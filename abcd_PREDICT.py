import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pyreadr
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import statsmodels.api as sm
from pathlib import Path
import random
import glob
from functools import reduce
import os.path
from os import path

# ============= WIP script for ML prediction models in ABCD =================

# first input class data from mplus output > merge with original NDAR IDs
# read in, select, clean and normalise all variables [0,1]
# build design matrix
# impute missing data
# split test and train > run ML models
# evaluate feature importance

# class data
abcd_4k = pd.read_table('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/'
                        'Edinburgh/gmm/gmm_abcd/mplus_data/4k_probs_abcd_cbcl.txt', delim_whitespace=True, header=None)
abcd_4k.columns = ['y0','y1','y2','y3','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','class','id']
abcd_4k = abcd_4k[['id','class']] #subsetting
abcd_4k.replace('*', np.nan) #replace missing

# class data has unique id's only > merge with anth to align NDAR id's
os.chdir('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/gmm/gmm_abcd')

# if running script on eddie
# from abcd_PREP.py import get_dataframes
# df1, df2 = get_dataframes()

# if local
anth = pd.read_csv("/Volumes/igmm/GenScotDepression/users/poppy/abcd/abcd_anth.csv")

abcd_4k = pd.merge(abcd_4k, anth, on='id')

# ================================VARIABLES=====================================
# make df with all vars, clean all vars (X), append class col (Y)

data = abcd_4k
data = data.rename(columns={'og_id': 'src_subject_id', 'sex': 'SEX', 'time': 'eventname'})
print(data) # this should include mapped numeric and og id's and class enumeration


# change to var storage area
os.chdir('/Volumes/igmm/GenScotDepression/data/abcd/release4.0/iii.data/')
# all files containing VOIs
rds = ['fhxp102.rds','dibf01.rds','abcd_mhp02.rds','abcd_cbcls01.rds','abcd_asrs01.rds', 'abcd_ypdms01.rds',
       'abcd_ppdms01.rds', 'abcd_yssbpm01.rds', 'abcd_ssphp01.rds','abcd_sds01.rds', 'stq01.rds', 'abcd_sscep01.rds',
       'abcd_fes01.rds', 'abcd_fhxssp01.rds', 'abcd_mhy02.rds', 'abcd_ksad01.rds' ]

datasets = [data] # list of datasets with the base data in
count = 1 # 1 df already in list

# get all dataframes from rds list and append using os.walk
for root, dirs, files in os.walk(".", topdown=False):
   for file in files:
      if file in rds:
          df = pyreadr.read_r(os.path.join(root, file))
          df = df[None]  # changes from dict
          datasets.append(df)
          count = count + 1

# extract VOIs: binary, continuous, demographic
b_vars = ['kbi_p_c_bully','fam_history_q6d_depression','fam_history_q6a_depression','famhx_4_p', 'menstrualcycle4_y',
          'menstrualcycle4_p', 'fes_youth_q6','fes_youth_q1', 'famhx_ss_moth_prob_alc_p', 'famhx_ss_fath_prob_alc_p',
          'ksads_21_921_p', 'ksads_21_922_p']
c_vars = ['screentime_1_wknd_hrs_p', 'fes_p_ss_fc', 'fes_p_ss_fc_pr', 'macv_p_ss_fs', 'macv_p_ss_fo', 'macv_p_ss_isr',
          'psb_p_ss_mean', 'asr_scr_depress_r', 'asr_scr_anxdep_r', 'peq_ss_overt_victim', 'cbcl_scr_dsm5_anxdisord_r','sds_p_ss_total']
dem_vars = ['src_subject_id','class','eventname','age','SEX']
binary = dem_vars + b_vars
cont = dem_vars + c_vars
all_vars = dem_vars + b_vars + c_vars

features = b_vars + c_vars

# to prevent duplicate cols
for df in datasets:
    df.drop(['interview_date','interview_age','sex'], axis=1, inplace=True, errors='ignore')
    df['eventname'] = df['eventname'].replace(['baseline_year_1_arm_1', '6_month_follow_up_arm_1', '1_year_follow_up_y_arm_1',
                                               '18_month_follow_up_arm_1', '2_year_follow_up_y_arm_1', '30_month_follow_up_arm_1',
                                               '3_year_follow_up_y_arm_1'], [0, 0.5, 1, 1.5, 2, 2.5, 3],)
    df['eventname']=df['eventname'].astype(float) # all floats for merging
    df['src_subject_id']=df['src_subject_id'].astype(object) # all objects for merging

print('done - datasets is a list of dataframes')


# =================================================================================================================
# ======================================== data frame joining and cleaning ========================================
# ==================================================================================================================


# merging all datasets on id and time point
all = reduce(lambda left,right: pd.merge(left,right,on=['src_subject_id','eventname'], how='outer'), datasets)

# filling NaNs in data cols
# Create a dictionary for each
id_class_dict = all.dropna(subset=['class']).set_index('src_subject_id')['class'].to_dict()
id_sex_dict = all.dropna(subset=['SEX']).set_index('src_subject_id')['SEX'].to_dict()
id_event_age_dict = all.dropna(subset=['age']).set_index(['src_subject_id', 'eventname'])['age'].to_dict() # age changes with time

# Use the dictionary to fill NaN values in specified column
all['class'] = all['class'].fillna(all['src_subject_id'].map(id_class_dict))
all['SEX'] = all['SEX'].fillna(all['src_subject_id'].map(id_sex_dict))
all['age'] = all.apply(lambda row: id_event_age_dict.get((row['src_subject_id'], row['eventname']), row['age']), axis=1)


# useful code to check column values match the unique id's:
'''class_counts = df.groupby('src_subject_id')['class'].nunique()

# Check if any 'src_subject_id' values have more than one unique 'class' value
if (class_counts > 1).any():
    print("Warning: 'class' values are not consistent for all 'src_subject_id' values.")
else:
    print("All 'src_subject_id' values have the same 'class' value.")'''

# remove the numeric id col used in mplus
all = all.drop(columns=['id'])
column_to_move = all.pop("src_subject_id") # moving id to first col
df = all.copy() # to prevent fragmented dataframe
df.insert(0, "src_subject_id", column_to_move) # insert column with insert(location, column_name, column_value)

df_bkup = df

# =================================================================================================================
# ================================= ADD GENETIC VARS =============================================================

# read in PRS sccores from .best file

prs1 = pd.read_csv('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/prs/prs_abcd_OUT/abcd_mdd_prs_test_230301.best', sep='\s+')
prs1 = prs1[['IID', 'PRS']]
prs1 = prs1.rename(columns={'IID':'src_subject_id', 'PRS':'prs_mdd'})
df = pd.merge(df, prs1, on='src_subject_id', how='left')

# list of PRSs
prs_vars = ['prs_mdd']

# add to all_vars and c_vars lists
all_vars = all_vars + prs_vars
c_vars = c_vars + prs_vars

# ================================= MAKING DESIGN MATRIX ===================================

# useful code to make table
df['class'].fillna('', inplace=True)  # fill empty values in 'class' column with empty string
grouped = df.groupby(['eventname', 'class'])['class'].count().unstack(fill_value=0)
result = grouped.astype(int) # shows 2 NaNs at event 0 that were excluded in mplus
print(result)


# select only vars we want
df = df[all_vars]
# now recode all binary variables so they are restricted to [0,1,NaN]
df.loc[:, b_vars] = df[b_vars].replace({'yes': 1, 'no': 0}).mask(~df[b_vars].isin([0,1]))
# now to normalise c_vars in range [0,1] or NaN
df.loc[:, c_vars] = df[c_vars].astype(float).apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.min() != x.max() else np.nan)


# check if features are between 0 and 1 or NaN
for feature in features:
    if df[feature].notnull().all():
        if (df[feature] >= 0) & (df[feature] <= 1).all():
            pass
        else:
            print(f"Some non-NaN values in {feature} are not between 0 and 1.")
    else:
        print(f"{feature} contains NaN values.")


# ========================================================================================
# ============================= impute missing values ====================================
# ========================================================================================

# once all vars are selected
# options are mean or median (same column) or KNNn (based on values of similar rows)

# ================================ testing single var ====================================

# class 2 is stable low so reference class in ABCD, change to 0 for reference
# c1 high, c3 decreasing, c4 increasing

# using statsmodels

# Filter out rows with missing values in 'prs_mdd' and 'class' columns
df2 = df.dropna(subset=['prs_mdd', 'class'])
df2.loc[:, 'class'] = df2['class'].replace(2.0, 0.0) # replace with 0 for ref level
x = df2['prs_mdd']
y = df2['class']
X = sm.add_constant(x)
model = sm.MNLogit(y, X)
result = model.fit()
print(result.summary())


# using scikitlearn

filtered_df = df.dropna(subset=['prs_mdd', 'class'])
filtered_df = filtered_df.copy()
filtered_df.loc[:,'class'] = filtered_df['class'].replace(2.0, 0.0)

x = np.array(filtered_df['prs_mdd']).reshape(-1, 1)
y = np.array(filtered_df['class'])

clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
clf.fit(x, y)

print("Coefficients: ", clf.coef_)
print("Intercepts: ", clf.intercept_)
print("Odds Ratios: ", np.exp(clf.coef_))
print("Classes: ", clf.classes_)



# ================= Classification models ==================
# Separate input and target variables, split
X = df.drop(['class', 'src_subject_id','class','eventname','age'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# multinomial LR
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))
r_sq = clf.score(x, y)
print('coefficient of determination:', r_sq)
y_pred = clf.predict(x)
print('Predicted response:', y_pred, sep='\n')
OR = np.exp(clf.coef_)
print('odds ratio', OR)
clf.coef_
clf.classes_

