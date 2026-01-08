# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 15:56:42 2025

@author: shang
"""


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA,LatentDirichletAllocation
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import os
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.size']=8
plt.rcParams['font.family']='Arial'
kubei=(71/255,75/255,87/255)
qiaoben=(227/255,165/255,103/255)
longyuan=(182/255,98/255,134/255)
yizhilai=(255/255,205/255,202/255)
zhuiming=(159/255,164/255,210/255)
senxia=(84/255,89/255,191/255)
tor_color=(187/255,155/255,230/255)
miq_color=(1,227/255,147/255)
mal_color=(197/255,91/255,51/255)
lan_color=(83/255,94/255,173/255)
dic_group_color={'Criollo':longyuan,'Nanay':kubei,'Amelonado':yizhilai,'Contamana':qiaoben,'Iquitos':zhuiming,'Purus':senxia,'Nacional':tor_color,
                 'Maranon':miq_color,'Guianna':mal_color,'Curaray':'red','Admixture':'grey'}
senxia_kubei_cmap=LinearSegmentedColormap.from_list("my_cmap", [kubei,senxia], N=20)
longyuan_kubei_cmap=LinearSegmentedColormap.from_list("my_cmap", [kubei,longyuan], N=20)
yizhilai_kubei_cmap=LinearSegmentedColormap.from_list("my_cmap", [kubei,yizhilai], N=20)
path='C:/Users/shang/liu_project/cocoa_pan_genome/03lda/'

#get the sv_multiple_dat and sv stat
sv_multiple_trait=pd.read_csv(path+'sv_multile_trait_final.csv',sep=',',index_col=0)
sv_multiple_trait=sv_multiple_trait.sort_index()
sv_stat=pd.read_csv(path+'200_pannel_final.csv',index_col=0)
sv_stat=sv_stat.loc[sv_stat.index.isin(sv_multiple_trait.index.tolist())].sort_index()
sv_stat=sv_stat.T

#transfer the lda effect frequency matrix
lda_pre=np.matmul(sv_stat,sv_multiple_trait)
lda_pre=lda_pre - lda_pre.min()

#start LDA analysis
dic_lda_result={}
for k in range(2,10):
    lda_pos=LatentDirichletAllocation(n_components=k,random_state=0)
    lda_pos.fit(lda_pre)
    dic_lda_result[k]=(lda_pos.score(lda_pre),lda_pos.perplexity(lda_pre))
#number of topics is 2 according to perplexity
lda_pos=LatentDirichletAllocation(n_components=2,random_state=0)
lda_pos.fit(lda_pre)
#write the topic-accession and topic-trait matrix
theta_pos=lda_pos.transform(lda_pre)
phi_pos=lda_pos.components_
theta_pos_norm = theta_pos / theta_pos.sum(axis=1, keepdims=True)
phi_pos_norm = phi_pos / phi_pos.sum(axis=0, keepdims=True)
theta_pos_norm=pd.DataFrame(theta_pos_norm)
phi_pos_norm=pd.DataFrame(phi_pos_norm)
theta_pos_norm.index=lda_pre.index
phi_pos_norm.columns=lda_pre.columns
phi_pos_norm=phi_pos_norm.T
theta_pos_norm.columns=['Topic1','Topic2']
phi_pos_norm.columns=['Topic1','Topic2']
theta_pos_norm.to_csv(path+'pos_lda_theta.csv')        
phi_pos_norm.to_csv(path+'pos_lda_phi.csv')     

#assigned the group information
dat_200_category=pd.read_csv(path+'../02sv/200_pannel/genetic_group.csv',index_col=0)
dic_group={}
for group in dat_200_category.columns:
    tmp=dat_200_category.loc[dat_200_category[group]==1].index.tolist()
    for i in tmp:
        dic_group[i]=group
theta_pos_norm['group']=[dic_group[i] for i in theta_pos_norm.index]
theta_pos_norm=theta_pos_norm.sort_values(['group','Topic1','Topic2'])

#plot the trait-topic matrix and accessions-topic matrix
fig,ax1=plt.subplots(1,1,figsize=(3,1.5),dpi=600)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.bar(np.arange(0,phi_pos_norm.shape[0]),phi_pos_norm['Topic1'],color=kubei)
ax1.bar(np.arange(0,phi_pos_norm.shape[0]),phi_pos_norm['Topic2'],color=longyuan,bottom=phi_pos_norm['Topic1'])
ax1.set_xticks(np.arange(0,phi_pos_norm.shape[0]),phi_pos_norm.index,rotation=45,ha='right',fontsize=6)
ax1.set_ylabel('Topic score',fontsize=6)
fig.tight_layout()
fig.savefig(path+'pos_topic_score_trait.svg')
fig.clf()

fig,ax1=plt.subplots(1,1,figsize=(3,1.5),dpi=600)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.bar(np.arange(0,theta_pos_norm.shape[0]),theta_pos_norm['Topic1'],color=[dic_group_color[i] for i in theta_pos_norm['group']])
ax1.bar(np.arange(0,theta_pos_norm.shape[0]),-theta_pos_norm['Topic2'],color=[dic_group_color[i] for i in theta_pos_norm['group']])
ax1.hlines(0,200,0,color='black')
ax1.hlines(0.5,200,0,color='black',linestyle='dashed',linewidth=1)
ax1.hlines(-0.5,200,0,color='black',linestyle='dashed',linewidth=1)
ax1.set_xticks([])
ax1.set_xlabel('200 cocoa accessions')
ax1.set_yticks([-1,0,1],['1','0','1'])
ax1.set_ylabel('Topic2 score/Topic1 score',fontsize=6)
fig.tight_layout()
fig.savefig(path+'pos_topic_score_accession.svg')
fig.clf()

row_colors=theta_pos_norm['group'].map(dic_group_color)
g = sns.clustermap(theta_pos_norm.iloc[:,:-1],colors_ratio=(0.1,0.02), cmap=longyuan_kubei_cmap,row_colors=row_colors, figsize=(1.5,6),dendrogram_ratio=(0.15,0.06),cbar_kws={'shrink':0.5} )
g.ax_heatmap.yaxis.set_visible(False)
g.fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)
g.savefig(path+"/sv_10_topic_heatmap.svg", dpi=600, bbox_inches="tight")
g.fig.clf()

from scipy.cluster.hierarchy import linkage,fcluster
Z=g.dendrogram_row.linkage
Z_index=g.data2d.index
tree_label=fcluster(Z,t=0.3,criterion='distance')
tree_label=pd.DataFrame(tree_label,index=Z_index)
tree_label['group']=[dic_group[i] for i in tree_label.index]
theta_pos_norm_group=theta_pos_norm.T
cluster_sample=[i for i in tree_label.index]
theta_pos_norm_group=theta_pos_norm_group[cluster_sample]
theta_pos_norm_group=theta_pos_norm_group.T
theta_pos_norm_group.to_csv(path+'domestication_group.csv')

#add the four groups manually according to the 2 topic values.
group_result=pd.read_csv(path+'domestication_group.csv',index_col=0)
group1=group_result.loc[group_result['Domestication group']=='group1']
group2=group_result.loc[group_result['Domestication group']=='group2']
group3=group_result.loc[group_result['Domestication group']=='group3']
group4=group_result.loc[group_result['Domestication group']=='group4']

#plot the topic pattern
fig,(ax1,ax2,ax3,ax4)=plt.subplots(1,4,figsize=(6,1.5),dpi=600,sharey=True)
num=0
for group_,ax in zip([group4,group3,group2,group1],[ax1,ax2,ax3,ax4]):
    group_mean=group_.iloc[:,:2].mean(axis=0)
    group_std=group_.iloc[:,:2].std(axis=0)
    num+=1
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(np.arange(2), group_mean, color='black', marker='o', label='Mean')
    ax.fill_between(np.arange(2), group_mean - group_std,  group_mean + group_std, color='black', alpha=0.2, label='±1 SD')
    ax.set_xticks(np.arange(2),[i for i in group_mean.index],rotation=45,ha='right')
    ax.set_title('Group%d'%(5-num))
    if num==1:
        ax.set_ylabel('Topic score')
    else:
        ax.set_ylabel('')
fig.tight_layout()
fig.savefig(path+'topic_score_plot.svg')
fig.clf()

#calculate the distribution of 10 genetic groups among 4 domestication groups
def summarize_groups(df):
    count_table = df.groupby(["Domestication group", "group"]).size().unstack(fill_value=0)
    norm_table = count_table.div(count_table.sum(axis=0), axis=1)
    return count_table, norm_table
group_result_count = summarize_groups(group_result)
norm_group_count=group_result_count[1]
group_result_count[0].to_csv(path+'dom_group_stat.csv')
g = sns.clustermap(norm_group_count,colors_ratio=(0.05,0.1), cmap=yizhilai_kubei_cmap, figsize=(6,2),dendrogram_ratio=(0.1,0.1),cbar_kws={'shrink':0.5} )
#g.ax_heatmap.yaxis.set_visible(False)
g.fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.05)
g.savefig(path+"/group_dom_heatmap.svg", dpi=600, bbox_inches="tight")
g.fig.clf()