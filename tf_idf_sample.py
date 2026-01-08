# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 13:52:01 2026

@author: shang
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import pearsonr
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import permutation_test
from scipy.stats import chi2_contingency

plt.rcParams['font.size']=8
plt.rcParams['font.family']='Arial'
path='C:/Users/shang/liu_project/cocoa_pan_genome/'
def tf_idf(data,target,background):
    dat_target=data[target]
    dat_background=data[background]
    tf_dat=dat_target.sum(axis=1)/dat_target.shape[1]
    idf_dat=np.log10(dat_background.shape[1]/(1+dat_background.sum(axis=1)))
    tf_idf=tf_dat*idf_dat
    fp_dat=pd.DataFrame()
    fp_dat['FP Score']=tf_idf
    fp_dat.index=dat_background.index
    fp_dat=fp_dat.sort_index()
    return fp_dat

#load the matrix
dat_sv_stat=pd.read_csv(path+'02sv/200_pannel/200_pannel_final.csv',index_col=0)
dat_sv_stat[dat_sv_stat>=1]=1
dat_geo=pd.read_csv(path+'03geo/geo_source.csv')

#cal fps
geos=['Ecuador','Trinidad','CostaRica']
dic_fp_geo={}
for geo in geos:
    dic_fp_geo[geo]=tf_idf(dat_sv_stat,dat_geo.loc[dat_geo['source']==geo]['accession'].tolist(),dat_geo.loc[~(dat_geo['source']==geo)]['accession'].tolist())
dat_fp_geo=pd.DataFrame()
for i in geos:
    dat_fp_geo[i]=[j for j in dic_fp_geo[i]['FP Score']]
dat_fp_geo.index=dic_fp_geo[i].index
dat_fp_geo.to_csv(path+'03geo/dat_geo_fp.csv')