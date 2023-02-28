import numpy as np
import os
import pandas as pd
import seaborn as sns
import pyreadr
import matplotlib.pyplot as plt
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

# exploring whether BPM internalising scale is an appropriate self-report scale for trajectories

# ============= BPM data ===============
df = pyreadr.read_r('/Volumes/igmm/GenScotDepression/data/abcd/release4.0/iii.data/Mental_Health/abcd_yssbpm01.rds')
df = df[None]
df = df[['src_subject_id', 'interview_age', 'eventname','bpm_y_scr_internal_r', 'bpm_y_ss_internal_mean']] # also externalising
df.columns = ['id', 'age', 'time','int_r', 'int_mean']
df['og_id'] = df.loc[:, 'id'] # store og id in another col to merge with vars later on
original_ids = df['id'].unique()

# numeric id for mplus
random.seed(1)
while True:
    new_ids = {id_: random.randint(10_000_00, 99_999_999) for id_ in original_ids}  # 8 digit unique numeric id
    if len(set(new_ids.values())) == len(original_ids):
        # all the generated id's were unique
        break
    # otherwise this will repeat until they are
df['id'] = df['id'].map(new_ids)
df['id'].nunique() # check, 11,733

df['age'] = df['age'].div(12)

df['time'] = df['time'].replace(['baseline_year_1_arm_1', '6_month_follow_up_arm_1', '1_year_follow_up_y_arm_1',
                                               '18_month_follow_up_arm_1', '2_year_follow_up_y_arm_1', '30_month_follow_up_arm_1',
                                               '3_year_follow_up_y_arm_1'], [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

# saving for plotting age or time later
df2 = df[['id','age','time','int_r']]

# ========== now convert data to wide for mplus =============
# first keep only relevant cols
df = df[['id','time','int_r']]
data_wide = pd.pivot(df, index=['id'], columns='time', values='int_r')
data_wide[[0.5,1.0,1.5,2.0,2.5,3.0]] = data_wide[[0.5,1.0,1.5,2.0,2.5,3.0]].fillna('-9999') # replace NaNs with -9999 for mplus
filepath = Path('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/gmm/gmm_abcd/mplus_data/abcd_bpm_wide_python.txt')
filepath.parent.mkdir(parents=True, exist_ok=True)
data_wide.to_csv(filepath, header=False, sep='\t')  # save wide data for mplus, make sure tab separated


# ======= plot BPM trajectories ===========
#reading in class assignment data
bpm_4k = pd.read_table('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/gmm/gmm_abcd/'
                      'mplus_data/4k_probs_bpm.txt', delim_whitespace=True, header=None)
bpm_4k.columns = ['y0.5','y1','y1.5','y2','y2.5','y3','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','class','id']
bpm_4k = bpm_4k[['id','class']]
bpm_4k.replace('*', np.nan)

# plot classes against bpm raw scores
# df cols are id, time, int_r > want to merge class by id
abcd = pd.merge(bpm_4k, df2, on=["id"])

# create col that matches mean age to time point
mean_age = abcd.groupby('time')['age'].mean()
abcd['mean_age'] = abcd['time'].map(mean_age)

# Define a dictionary mapping each category to a specific color in the palette
palette = sns.color_palette("Set2")
color_map = {1.0: palette[1], 2.0: palette[3], 3.0: palette[2], 4.0: palette[0]}

# Create the lineplot with the custom colors
sns.lineplot(data=abcd, x='mean_age', y='int_r', hue='class', palette=color_map, legend=False)
plt.ylabel('Brief Problem Monitoring scale - internalising')
plt.title('ABCD n = 11,722')
plt.ylim(0,12)
plt.xlabel('age')

# Add the legend with the custom labels
handles, labels = plt.gca().get_legend_handles_labels()
labels = ['early high', 'increasing', 'decreasing', 'stable low']
plt.legend(handles, labels, title='Trajectory', loc='upper right')

plt.show()

# ============================================