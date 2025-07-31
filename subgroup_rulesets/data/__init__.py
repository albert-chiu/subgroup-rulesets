import os
import pandas as pd
import pickle

def load_jobcorps_data():
    """Load Job Crops data."""
    data_dir = os.path.dirname(__file__)
    temp = pd.read_csv(os.path.join(data_dir, 'jobcorps_cate.csv'))
    tau = temp['cate']
    Y = temp['Y']
    treat = temp['treat']
    train = temp['train']
    # rename
    cols_to_add = ['HASCHLD', 'ANY_ED1', 'GOT_ANYW', 'EVARRST1', 'GUILTY2', 'EVJAIL2', 'CURRJOB', "YR_WORK1"]
    temp.columns = [col + "_1" if col in cols_to_add else col for col in temp.columns]
    # discretize
    temp["INC_L3"] = (temp["PERS_INC"] == 1).astype(int)
    temp["INC_L6"] = (temp["PERS_INC"] <= 2).astype(int)
    temp["INC_L9"] = (temp["PERS_INC"] <= 3).astype(int)
    for q in [0.25, 0.5, 0.75]:
        q_val = temp["AGE"].quantile(q)
        col_name = f'AGE_L{int(q_val)}'
        temp[col_name] = (temp["AGE"] < q_val).astype(int)
    # cumulative 
    temp["WELFKID_NEVER"] = (temp["WELF_KID"] == 1).astype(int)
    temp["WELFKID_AMOCC"] = (temp["WELF_KID"] <= 2).astype(int)  # at most occassionally 
    temp["WELFKID_AMHALF"] = (temp["WELF_KID"] <= 3).astype(int)  # at most half
    # education variables
    temp["EDUC_ANYDEG"] = (temp[["HS_D", "GED_D", "VOC_D", "OTH_DEG"]].sum(axis=1) >= 1).astype(int)
    temp["EDUC_LHS"] = (temp["HGC"] < 12).astype(int)  # les than high school
    temp["EDUC_HS"] = (temp["HGC"] <= 12).astype(int)  # at least high school


    cols_to_drop = [col for col in temp.columns if "WELF" in col]
    cols_to_drop += ['AGE', 'HGC', 'GED_D', 'VOC_D', 'OTH_DEG', 'PERS_INC', 'cate', "Y", 'treat', 'train']
    X = temp.drop(columns=cols_to_drop)

    # split data
    X_train = X[train==1]
    X_test = X[train==0]
    tau_train = tau[train==1]
    tau_test = tau[train==0]

    return  X_train, X_test, tau_train, tau_test, Y, treat

def load_jobcorps_rulesets():
    """Load synthetic CATE example."""
    data_dir = os.path.dirname(__file__)
    with open(os.path.join(data_dir, 'jobcorps_rs.pkl'), 'rb') as f:
        return pickle.load(f)