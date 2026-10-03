#START OF CODE APPENDIX MADE BY RUBEN BERMAN 45735476========================================================================================================================================================================================================================================================================================================================================================================================================================
"""EXTRA TIPS
git pull                                  # 1. start of session: get partner's changes
# ... work, Ctrl+S as you go ...
git add .                                 # 2. stage everything you changed
git commit -m "Add t-SNE plot of ROA features"   # 3. snapshot with a message
git push                                  # 4. upload to GitHub
"""
# %%
#All library imports
import pandas as pd #Data frames and CSV importation
import seaborn as sns #Heatmaps, pairplots and histograms
import matplotlib.pyplot as plt #Plot titles, labels and displaying figures
import numpy as np #Numerical operations and the triangular heatmap masks
#Outlier detection models from the pyod library (found via a LinkedIn post)
from pyod.models.iforest import IForest #Isolation Forest: isolates outliers using random splits
from pyod.models.ecod import ECOD #Scores tail values using each variable's empirical CDF (ECDF)
from pyod.models.copod import COPOD #Like ECOD, but combines the variables with a copula
from sklearn.manifold import TSNE
from sklearn.preprocessing import QuantileTransformer, StandardScaler #Rank-based and standard scaling
from sklearn.cluster import KMeans #K-means clustering (earlier approach to the clustered variables)
from matplotlib.colors import LinearSegmentedColormap #Custom colour map for the correlation heatmaps
from matplotlib.ticker import FuncFormatter #Shows billion-scale axes as "2bn" instead of "1e10"

# %%
#House style shared by every figure in the report
SURFACE = '#fcfcfb' #Figure background
INK = '#0b0b0b' #Titles
INK_2 = '#52514e' #Axis labels
MUTED = '#898781' #Tick labels and source notes
GRID = "#e3e2db" #Gridlines
BLUE = "#386399" #Single colour for histograms and scatter points
SOURCE = 'Source: Taiwan Economic Journal, 1999-2009'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'figure.dpi': 120, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans'],
    'text.color': INK, 'axes.titlecolor': INK, 'axes.labelcolor': INK_2,
    'axes.titlesize': 13, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left', 'axes.titlepad': 12,
    'axes.labelsize': 10, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'legend.frameon': False,
})

#Axis formatter for the corrupted, billion-scale variables
BILLIONS = FuncFormatter(lambda v, _: f"{v/1e9:.0f}bn")

#Diverging colour map for correlations: red = negative, grey = none, blue = positive
CORR_CMAP = LinearSegmentedColormap.from_list('corr', ['#a32d2c', '#e34948', '#f0efec', '#2a78d6', '#104281'])

# %%
#Import the Taiwanese bankruptcy dataset
df = pd.read_csv('data.csv')
# %%
#Work on a copy so the original df stays untouched
df2 = df.copy()
# %%
#Net Income Flag is 1 for every company, so it has no variation
print(df[" Net Income Flag"])
#Predictors: drop Net Income Flag (no variation) and Bankrupt? (the response)
Xvar = df2.drop([" Net Income Flag","Bankrupt?"], axis=1)
# %%
#Response variable: 1 = bankrupt, 0 = not bankrupt
Yvar = df2["Bankrupt?"]
# %%

#Every genuine value in this dataset is rescaled to 0-1, and no value lies
#between 1 and 583,000, so anything >= 1000 is a corrupted entry.
#Count the corrupted values in each variable
Xbil = (Xvar>=1000).sum()
# %%
#Sort variables from most to fewest corrupted values
Xbil_sort = Xbil.sort_values(ascending=False)
# %%
#Proportion of rows that are corrupted in each variable (4 decimal places)
Xbil_summary = (round(Xbil_sort/(len(Xvar)),4))
# %%

#Variables with corrupted values in 50% or more of rows:
#too many to impute, so these are binary encoded further down
print(Xbil_summary[Xbil_summary>=0.5])
# %%
#Spearman is rank based, so the corrupted values cannot distort it (explained in report)
corr_matrix = Xvar.corr('spearman')


# %%
#The 3 variables most correlated with Total Asset Growth Rate
#(absolute correlation, excluding its correlation with itself)
topvars1 = (corr_matrix[' Total Asset Growth Rate'].drop(' Total Asset Growth Rate').abs()).sort_values(ascending=False).head(3)

#Pairplot of Total Asset Growth Rate against those 3 variables
g = sns.pairplot(Xvar[[' Total Asset Growth Rate'] + list(topvars1.index)], corner=True, diag_kind='kde',
                 plot_kws={'alpha': 0.4, 's': 12, 'color': BLUE, 'edgecolor': 'none'},
                 diag_kws={'color': BLUE, 'fill': True})
g.axes[-1, 0].xaxis.set_major_formatter(BILLIONS)
g.figure.suptitle("Total Asset Growth Rate against its three most correlated predictors",
                  x=0.02, ha='left', fontsize=13, fontweight='semibold')
g.figure.subplots_adjust(top=0.93)
g.figure.text(0.02, 0.005, SOURCE, fontsize=8, color=MUTED)
plt.show()
# %%
#The 3 variables most correlated with Cash Turnover Rate (excluding itself)
topvars2 =  (corr_matrix[' Cash Turnover Rate'].drop(' Cash Turnover Rate').abs()).sort_values(ascending=False).head(3)

#Pairplot of Cash Turnover Rate against those 3 variables
g = sns.pairplot(Xvar[[' Cash Turnover Rate'] + list(topvars2.index)], corner=True, diag_kind='kde',
                 plot_kws={'alpha': 0.4, 's': 12, 'color': BLUE, 'edgecolor': 'none'},
                 diag_kws={'color': BLUE, 'fill': True})
g.axes[-1, 0].xaxis.set_major_formatter(BILLIONS)
g.figure.suptitle("Cash Turnover Rate against its three most correlated predictors",
                  x=0.02, ha='left', fontsize=13, fontweight='semibold')
g.figure.subplots_adjust(top=0.93)
g.figure.text(0.02, 0.005, SOURCE, fontsize=8, color=MUTED)
plt.show()
# %%
#Histograms of the three variables that are over 50% corrupted,
#with the corrupted share worked out in each title
for var in [' Total Asset Growth Rate', ' Cash Turnover Rate', ' Research and development expense rate']:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(Xvar[var], bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.8, ax=ax)
    share = (Xvar[var] >= 1000).mean()
    ax.set_title(f"{var.strip()}: {share:.0%} of entries are corrupted")
    ax.xaxis.set_major_formatter(BILLIONS)
    ax.set_xlabel("Reported value (genuine values lie between 0 and 1, inside the first bar)")
    ax.set_ylabel("Number of companies")
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.01, 0.01, SOURCE, fontsize=8, color=MUTED)
    plt.show()
# %%

#Too many corrupted values to impute, so rather than throw these variables out,
#keep a binary flag: 1 = corrupted value, 0 = genuine value
Xvar['TAGR (encode)'] = (Xvar[' Total Asset Growth Rate']>1).astype(int)
Xvar['CTR (encode)'] = (Xvar[' Cash Turnover Rate']>1).astype(int)
Xvar['R&D (encode)'] = (Xvar[' Research and development expense rate']>1).astype(int)

# %%
Xvar2 = Xvar.copy()
#Drop the original versions now that their binary flags exist
Xvar2 = Xvar2.drop([' Cash Turnover Rate',' Total Asset Growth Rate',' Research and development expense rate'],axis=1)
# %%
#Working copy for the imputation and winsorising steps
Xvar3 = Xvar2.copy()

# %%
#Variables with 15-50% corrupted values (the first KNN candidates, output below)
Xbil_summary[(Xbil_summary<0.5) & (Xbil_summary>0.15)]
# %%

from sklearn.impute import KNNImputer
#KNN imputer: fills each missing value with the average of the 5 most similar companies
imputer = KNNImputer(n_neighbors=5)

"""Research and development expense rate    0.4375
Inventory Turnover Rate (times)          0.3336
Quick Asset Turnover Rate                0.3162
Operating Expense Rate                   0.3017
Current Asset Turnover Rate              0.1607
Fixed Assets Turnover Frequency          0.1565"""

# %%
#Variables with 1.5-50% corrupted values: these get KNN imputation
intermediate_vars = list(Xbil_summary[(Xbil_summary >= 0.015) & (Xbil_summary < 0.5)].index)

#Turn their corrupted values (> 1) into NaN so the imputer can fill them
Xvar3[intermediate_vars] = Xvar3[intermediate_vars].mask(Xvar3[intermediate_vars] > 1)
#Spearman correlations recomputed without the corrupted values (NaN pairs are skipped)
corr_nan = Xvar3.corr('spearman')
#Variables with no corrupted values, the only ones allowed as KNN neighbours
clean_vars = Xvar3.columns[(Xvar3 <= 1).all()]

for var in intermediate_vars:
    #The 4 clean variables most correlated with this variable
    topvars = corr_nan.loc[clean_vars, var].abs().sort_values(ascending=False).head(4)
    neighbour_vars = list(topvars.index)
    #Record which neighbours were used for each variable
    print(var, '<-', neighbour_vars)
    #Impute using only this variable and its 4 neighbours
    filled = imputer.fit_transform(Xvar3[[var] + neighbour_vars])
    #Column 0 is the imputed variable itself
    Xvar3[var] = filled[:, 0]
# %%
#Variables with under 1.5% corrupted values: few enough to winsorise
lower_vars = list(Xbil_summary[(Xbil_summary < 0.015) & (Xbil_summary > 0)].index)
# %%
for var in lower_vars:
    #The genuine (0-1) values of this variable
    clean = Xvar3[var][Xvar3[var] <= 1]
    #Cap everything at the 99th percentile of the genuine values
    Xvar3[var] = Xvar3[var].clip(upper=clean.quantile(0.99))
# %%
#Cleaned dataset: every corrupted value has now been flagged, imputed or winsorised
Xvar4 = Xvar3.copy()
# %%

#Spearman correlation matrix of the cleaned data
corr_matrix = Xvar4.corr('spearman')

#Heatmap of the correlation matrix. Only the lower triangle is shown,
#since the matrix is symmetric and the upper triangle repeats it
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, cmap=CORR_CMAP, vmin=-1, vmax=1, center=0, square=True,
            xticklabels=1, yticklabels=1, cbar_kws={'label': 'Spearman correlation', 'shrink': 0.6}, ax=ax)
ax.grid(False)
ax.tick_params(labelsize=5)
ax.set_title(f"Spearman correlation between all {len(corr_matrix)} predictors after cleaning")
plt.tight_layout(rect=(0, 0.03, 1, 1))
fig.text(0.01, 0.01, SOURCE, fontsize=8, color=MUTED)
plt.show()


# %%
#To avoid losing too much information, a variable is only dropped
#when its absolute correlation with another variable is above 0.95

#Keep each pair once (upper triangle, diagonal excluded) as a list of pairs
unique_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)).stack()
#Pairs with an absolute correlation above 0.95
unique_pairs_filtered = unique_pairs[unique_pairs.abs() > 0.95]

#From each highly correlated pair, keep one variable and drop the other.
#E.g. if A and B have a correlation of 0.97, keep A and drop B
# %%
variables_to_drop = []
#Start from the strongest correlations
unique_pairs_filtered = unique_pairs_filtered.sort_values(ascending=False)
for pair in unique_pairs_filtered.index:
    var1, var2 = pair
    #Only drop var2 if neither variable in the pair is already being dropped
    if var1 not in variables_to_drop and var2 not in variables_to_drop:
        variables_to_drop.append(var2)
# %%
#List of the dropped variables
print(pd.DataFrame(variables_to_drop))
#29/94 variables dropped
# %%
#Remove the highly correlated variables (30/94= 64 variables now)
Xvar_reduced = Xvar4.drop(variables_to_drop, axis=1)

#Split binary flag columns from continuous columns, so they can be scaled separately
flag_cols = [c for c in Xvar_reduced.columns if Xvar_reduced[c].nunique() <= 2]
cont_cols = Xvar_reduced.columns.drop(flag_cols)
# %%
#Copy of the reduced data for scaling
Xvar_scaled = Xvar_reduced.copy()

# %%
#Heatmap of the reduced data, to check the highly
#correlated variables have been removed
corr_matrix2 = Xvar_reduced.corr()

mask2 = np.triu(np.ones_like(corr_matrix2, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix2, mask=mask2, cmap=CORR_CMAP, vmin=-1, vmax=1, center=0, square=True,
            xticklabels=1, yticklabels=1, cbar_kws={'label': 'Pearson correlation', 'shrink': 0.6}, ax=ax)
ax.grid(False)
ax.tick_params(labelsize=6)
ax.set_title(f"Correlation between the {len(corr_matrix2)} predictors kept after removing pairs above 0.95")
plt.tight_layout(rect=(0, 0.03, 1, 1))
fig.text(0.01, 0.01, SOURCE, fontsize=8, color=MUTED)
plt.show()

#Overall much better, although some strong correlations remain between
#variables that look like duplicates under different names
# %%
#Fixed random seed so the results are reproducible
seed = 45735476
#Quantile transformer: rank-based scaling, so skewed variables
#and extreme values cannot dominate the distances
qt = QuantileTransformer(n_quantiles=100,random_state=seed,output_distribution='normal')

# %%
non_ratios = [
    ' Net Value Per Share (B)', ' Net Value Per Share (A)', ' Net Value Per Share (C)',
    ' Persistent EPS in the Last Four Seasons', ' Cash Flow Per Share',
    ' Revenue Per Share (Yuan ¥)', ' Operating Profit Per Share (Yuan ¥)',
    ' Per Share Net profit before tax (Yuan ¥)',
    ' Revenue per person', ' Operating profit per person', ' Allocation rate per person',
    ' Average Collection Days', ' No-credit Interval',
    ' Liability-Assets Flag',
]
assert all(c in Xvar.columns for c in non_ratios)
n_ratios = len(Xvar.columns) - len(non_ratios)
print(f"{n_ratios} of {len(Xvar.columns)} predictors are ratios ({n_ratios/len(Xvar.columns):.0%})")
# %%


#THIS IS THE DATA THAT WILL BE USED
#FOR THE t-SNE AND PCA
Xvarqt = Xvar_scaled.copy()
Xvarqt[cont_cols] = qt.fit_transform(Xvar_scaled[cont_cols])

#Export the data
#Xvarqt.to_csv('X_Clean.csv')
#Yvar.to_csv('Y_Clean.csv')
# %%
#============================================================================================================================================================================================================================================================================================================================================================================
#RUBEN'S t-SNE

<<<<<<< HEAD
Xvarqt_array = np.array(Xvarqt)
from sklearn.manifold import TSNE
k=150 #perplexity value, can be changed to see how it affects the results
X_reduced = TSNE(n_components=2, perplexity=k, 
     early_exaggeration=12.0, 
     learning_rate='auto', 
     max_iter=1000, 
     n_iter_without_progress=300, 
     min_grad_norm=1e-07, 
     metric='euclidean', 
     metric_params=None, 
     init='random', 
     verbose=2, 
     random_state=45735476, 
     method='barnes_hut', 
     angle=0.8, 
     n_jobs=-1).fit_transform(Xvarqt_array)


from sklearn.cluster import DBSCAN

y  = np.ravel(Yvar)
c1, c2 = X_reduced[:, 0], X_reduced[:, 1]
clusters = DBSCAN(eps=3, min_samples=10).fit_predict(X_reduced)

g = sns.JointGrid(height=9)

# Contours: one colour per cluster (non-bankrupt only, skipping the DBSCAN noise label -1)
mask = (y == 0) & (clusters != -1)
sns.kdeplot(x=c1[mask], y=c2[mask], hue=clusters[mask], palette='Set2',
            levels=6, fill=True, alpha=0.5, legend=False, ax=g.ax_joint)

# Bankrupt points on top
g.ax_joint.scatter(c1[y == 1], c2[y == 1], s=60, color='red',
                   edgecolors='black', linewidths=0.8, zorder=3, label='Bankrupt')

# Histogram along each axis, split by class
pal = {0: '#2a78d6', 1: 'red'}
sns.histplot(x=c1, hue=y, palette=pal, bins=50, stat='density', common_norm=False,
             element='step', legend=False, ax=g.ax_marg_x)
sns.histplot(y=c2, hue=y, palette=pal, bins=50, stat='density', common_norm=False,
             element='step', legend=False, ax=g.ax_marg_y)

g.set_axis_labels('t-SNE Component 1', 't-SNE Component 2', fontsize=14)
g.ax_joint.legend(loc='upper right')
g.figure.suptitle(f't-SNE Visualization of Bankruptcy Data, perplexity = {k}', fontsize=16, y=1.02)
plt.show()



#FROM: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
#https://github.com/scikit-learn/scikit-learn/blob/afe5bcd80/sklearn/manifold/_t_sne.py
=======
# Xvarqt_array = np.array(Xvarqt)
# from sklearn.manifold import TSNE
# X_reduced = TSNE(n_components=2, perplexity=30.0,
#      early_exaggeration=12.0,
#      learning_rate='auto',
#      max_iter=1000,
#      n_iter_without_progress=30,
#      min_grad_norm=1e-07,
#      metric='euclidean',
#      metric_params=None,
#      init='pca',
#      verbose=0,
#      random_state=None,
#      method='barnes_hut',
#      angle=0.5,
#      n_jobs=None).fit_transform(Xvarqt_array)
#
#
#
# #FROM: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
# #https://github.com/scikit-learn/scikit-learn/blob/afe5bcd80/sklearn/manifold/_t_sne.py
#

>>>>>>> 079669694ef375dfcc7e993920c19c2258c3fac9
# %%




#==============================================================================================================================================================================================================================================================================================================================================================================
#MUHAMMAD'S PCA


#=============================================================================================================================================================================================================================================================================================================================================================================

# %%
