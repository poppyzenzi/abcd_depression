import numpy as np
import os
import pandas as pd
import pyreadr
from pathlib import Path
import random

# ========= PREP script to assign numeric ID's to abcd data  ========
# ========= then convert to wide for MPlus LCGA analysis ============
# when running on EDDIE, file paths will change to datastore locations

#first use anthropomorphic data to assign unique numeric ID's
anth = pyreadr.read_r('/Volumes/igmm/GenScotDepression/data/abcd/release4.0/iii.data/Physical_Health/abcd_ant01.rds')
anth = anth[None] # let's check what objects we have: there is only None
anth = anth[['src_subject_id','eventname', 'interview_age', 'sex']]
anth.columns = ['id','time','age','sex']
anth['time'] = anth['time'].replace(['baseline_year_1_arm_1','1_year_follow_up_y_arm_1',
                                 '2_year_follow_up_y_arm_1','3_year_follow_up_y_arm_1'],
                                [0,1,2,3])
anth['sex'] = anth['sex'].replace(['F','M'], [1,0])
anth['age'] = anth['age'].div(12)
anth['id'].nunique() # check 11,876
anth['og_id'] = anth.loc[:, 'id'] # keep old id

original_ids = anth['id'].unique()

random.seed(1)
while True:
    new_ids = {id_: random.randint(10_000_00, 99_999_999) for id_ in original_ids}  # 8 digit unique numeric id
    if len(set(new_ids.values())) == len(original_ids):
        # all the generated id's were unique
        break
    # otherwise this will repeat until they are
anth['id'] = anth['id'].map(new_ids)
anth['id'].nunique() # check 11,876

# to ensure seed it working, check mplus output ids with ascending anth ids
anth.sort_values(by=['id'])

# ============================  CBCL data  =============================
df = pyreadr.read_r('/Volumes/igmm/GenScotDepression/data/abcd/release4.0/iii.data/Mental_Health/abcd_cbcls01.rds')
df = df[None]
df = df[['src_subject_id','eventname','cbcl_scr_dsm5_depress_r']]
df.columns = ['og_id','time','dep']
df['time'] = df['time'].replace(['baseline_year_1_arm_1', '1_year_follow_up_y_arm_1',
                                     '2_year_follow_up_y_arm_1', '3_year_follow_up_y_arm_1'],
                                    [0, 1, 2, 3])
df['og_id'].nunique() # check 11,876. Should be same but we merge with anth to be certain

# merging into long format df
data = pd.merge(anth, df, on=["og_id","time"])

def get_dataframes(): # to import to other scripts - doesn't work from console
    df1 = anth # [id, time, age, sex, NDAR id]
    df2 = data # [id, time, age, sex, NDAR id, cbcl score]
    return df1, df2


# saving for local use
anth.to_csv("/Volumes/igmm/GenScotDepression/users/poppy/abcd/abcd_anth.csv", index=False)
data.to_csv("/Volumes/igmm/GenScotDepression/users/poppy/abcd/abcd_cbcl.csv", index=False)

# ==========now convert data to wide for mplus=============
# first keep only relevant cols
data = data[['id','time','dep']]
data_wide = pd.pivot(data, index=['id'], columns='time', values='dep') #  11,876 rows
data_wide[[0,1,2,3]] = data_wide[[0,1,2,3]].fillna('-9999') # replace NaNs with -9999 for mplus
filepath = Path('/Users/poppygrimes/Library/CloudStorage/OneDrive-UniversityofEdinburgh/Edinburgh/gmm/gmm_abcd/mplus_data/abcd_cbcl_wide_python.txt')
filepath.parent.mkdir(parents=True, exist_ok=True)
data_wide.to_csv(filepath, header=False, sep='\t')  # save wide data for mplus, make sure tab separated

