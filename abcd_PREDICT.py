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

# ============= WIP script for ML prediction models in ABCD =================

# first input class data from mplus output > merge with original NDAR IDs
# read in, select, clean and normalise all variables [0,1]
# build design matrix
# impute missing data
# split test and train > run ML models
# evaluate feature importance

# class data
abcd_4k = pd.read_table('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/'
                        'Edinburgh/gmm/gmm_abcd/mplus_data/4_class_probs_abcd_py.txt', delim_whitespace=True, header=None)
abcd_4k.columns = ['y0','y1','y2','y3','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','class','id']
abcd_4k = abcd_4k[['id','class']] #subsetting
abcd_4k.replace('*', np.nan) #replace missing

# class data has unique id's only > merge with anth to align NDAR id's
from abcd_PREP import get_dataframe
df = get_dataframe()

abcd_4k = pd.merge(abcd_4k, df, on='id')

# ================================VARIABLES=====================================

# make df with all vars, clean all vars (X), append class col (Y)
# one binary frame and one continuous. continuous scale between 0 and 1?
data = abcd_4k

data = data.rename(columns={'og_id': 'src_subject_id', 'sex': 'SEX', 'time': 'eventname'})

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

# extract VOIs
b_vars = ['kbi_p_c_bully','fam_history_q6d_depression','fam_history_q6a_depression','famhx_4_p', 'menstrualcycle4_y',
          'menstrualcycle4_p', 'fes_youth_q6','fes_youth_q1', 'famhx_ss_moth_prob_alc_p', 'famhx_ss_fath_prob_alc_p',
          'ksads_21_921_p', 'ksads_21_922_p' ]  # binary vars

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

# merging all datasets on id
all = reduce(lambda left,right: pd.merge(left,right,on=['src_subject_id','eventname'], how='outer'), datasets)
all['src_subject_id'].nunique() # again check 11876 unique id's
all = all.drop(columns=['id']) # remove the numeric id col used in mplus
column_to_move = all.pop("src_subject_id") # moving id to first col
df = all.copy() # to prevent fragmented dataframe
df.insert(0, "src_subject_id", column_to_move) # insert column with insert(location, column_name, column_value)

# ================================= MAKING DESIGN MATRIX ===================================

# useful code to make table
df['class'].fillna('', inplace=True)  # fill empty values in 'class' column with empty string
grouped = df.groupby(['eventname', 'class'])['class'].count().unstack(fill_value=0)
result = grouped.astype(int) # shows 2 NaNs at event 0 that were excluded in mplus
print(result)


df = df[all_vars] # select only vars we want

# now recode all binary variables so they are restricted to [0,1,NaN]
df[b_vars] = df[b_vars].replace({'yes': 1, 'no': 0})
df[b_vars] = df[b_vars].mask(~df[b_vars].isin([0,1]))

# now to normalise c_vars in range [0,1] or NaN
df[c_vars] = df[c_vars].astype(float)
df[c_vars] = (df[c_vars] - df[c_vars].min()) / (df[c_vars].max() - df[c_vars].min())

# check if features are between 0 and 1 or NaN
for feature in features:
    if df[feature].dropna().between(0, 1).all():
        pass
    else:
        print(f"Some non-NaN values in {feature} are not between 0 and 1.")

# vars ready

# will also want to append columns with polygenic scores once made

# ======================= here you would impute missing values ================
# options are mean or median (same column) or KNNn (based on values of similar rows)


# mulitnom logistic regression
# Separate input and target variables, split
X = df.drop(['class', 'src_subject_id','class','eventname','age'], axis=1)

y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
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

