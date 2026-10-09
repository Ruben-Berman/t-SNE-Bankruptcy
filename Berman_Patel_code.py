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
SURFACE = '#000000' #Figure background (black)
INK = '#ffffff' #Titles
INK_2 = '#f2f2f0' #Axis labels
MUTED = '#c9c8c3' #Tick labels and source notes
GRID = "#2e2e2b" #Gridlines
BLUE = "#6b9be0" #Single colour for histograms and scatter points (lighter so it shows on black)
YELLOW = '#e3a33b' #Second series colour
RED = '#ef5350' #Bankrupt / third series colour (brighter than pure dark red)
SOURCE = 'Source: Taiwan Economic Journal, 1999-2009'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'figure.dpi': 120, 'savefig.dpi': 300,
    'font.family': 'sans-serif', 'font.sans-serif': ['Segoe UI', 'DejaVu Sans'],
    'text.color': INK, 'axes.titlecolor': INK, 'axes.labelcolor': INK_2,
    'axes.titlesize': 13, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left', 'axes.titlepad': 12,
    'axes.labelsize': 10, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.8,
    'axes.prop_cycle': plt.cycler(color=[BLUE, YELLOW, RED, '#5cc98a', '#b48be8', '#4fc3d9']), #Bright defaults
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'legend.frameon': False,
})

#Axis formatter for the corrupted, billion-scale variables
BILLIONS = FuncFormatter(lambda v, _: f"{v/1e9:.0f}bn")

#Diverging colour map for correlations: red = negative, grey = none, blue = positive
#(bright ends and a dark grey middle so it reads on the black background)
CORR_CMAP = LinearSegmentedColormap.from_list('corr', ['#ff6b6b', '#b8403f', '#3a3a38', '#3f7fd0', '#8fc1ff'])

# %%
#Folder this script is in, so data.csv is found wherever the code is run from
#(__file__ when run as a script, __vsc_ipynb_file__ in the VS Code interactive window,
#otherwise the current folder; the first one that actually contains data.csv is used)
from pathlib import Path
folders = [Path(globals()[name]).resolve().parent
           for name in ('__file__', '__vsc_ipynb_file__') if name in globals()] + [Path.cwd()]
HERE = next((f for f in folders if (f / 'data.csv').exists()), Path.cwd())
print('Reading data from:', HERE)

#Import the Taiwanese bankruptcy dataset
df = pd.read_csv(HERE / 'data.csv')
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
#==============================================================================================================================================================================================================================================================================================================================================================================
#OUTLIER DETECTION (on the raw data, before any cleaning: this is how the corrupted values were found)

cr = Yvar.mean() #Share of bankrupt companies
print(f'Contamination rate to \napply: {round(100*cr,2)}%')
# %%


# %% 07 | Model 1 - Isolation Forest
#Set seed for reproducability
seed = 45735476
#Fit isolation forest, unsupervised
clf = IForest(contamination = cr,random_state=seed)
clf.fit(Xvar)

# %% 08 | Model 2 - ECOD with the same contamination rate

clf2 = ECOD(contamination = cr)
clf2.fit(Xvar)

# %% 09 | Model 3 - COPOD with the same contamination rate
#Same for COPOD
clf3 = COPOD(contamination=cr)
clf3.fit(Xvar)

# %% 10 | Predicted labels from all three models
#Our labels are either 1
#for outlier or 0 for inlier
Ypred = clf.labels_
Ypred2 = clf2.labels_
Ypred3 = clf3.labels_

# %% 11 | Consensus mask - rows all three models agree on
#Stacking them together
#going to find out where does
#each ML all flag an anomaly at once
#To reduce any bias
anomalies = (
    (Ypred==1) &
    (Ypred2==1) &
    (Ypred3==1)
)

# %% 12 | Subsetting the consensus anomalies
#X and Y variables for where all
#3 models agreed that row_i is a row
#that must be flagged as a anomaly
Xanon = Xvar[anomalies]
Yanon = Yvar[anomalies]

# %% 13 | Summary of the flagged rows
print(f"Out of the total {len(Yanon)} rows\nflagged as anomalies  \n{np.sum(Yanon)} are positive bankruptcies and {len(Yanon)-np.sum(Yanon)}\nare negative for any bankruptcy,\nthese {len(Yanon)-np.sum(Yanon)} 'outliers' must be checked for\npotential mis-imputation, and the other\n{np.sum(Yanon)} rows must be scrutinised as to what\nexactly are causing the bankruptcies.")

#How many rows to every
#ML flag as anomaly
#turns out to be 2%
#close to original class
#imbalance being 3%

print(f"We got {len(Xanon)} rows\nflagged as anomalies out of the total \n{len(Xvar)} rows in the data. which is {round(100*len(Xanon)/len(Xvar),2)}%")

# %%
mu = Xvar.mean()
s = Xvar.std()
#The absolute Z score, because it is
#A question of now whether the movement
#in centrality is caused by an extremely 
# large outlier or extremely small outlier
#But whether there is an outlier at all
# |Z| =|(X-mu)/sigma|
Zscoreord = np.abs((Xvar - mu)/s)
#Z score for anomaly/outlier vals
ordanon = Zscoreord[anomalies].mean()

# %% Absolute Z scores, inlier group mean
#Z score for non-anomaly vals
ordinline = Zscoreord[~anomalies].mean()

Zlistfinal_unsorted_df = (ordanon/ordinline).reset_index()
Zlistfinal_unsorted_df.columns = ['Variable', 'Zratio']
Zlistfinal_df = Zlistfinal_unsorted_df.sort_values(by='Zratio', ascending=False)
# %% 
#plot Zlistfinal_df to see where there is a strong drop in Z scores
#And it is flat within 4 variables, entailing potential duplication
plt.figure(figsize=(12,6))
plt.bar(Zlistfinal_df['Variable'], Zlistfinal_df['Zratio'], color=BLUE)
plt.title("Z ratio of anomaly vs inlier group means")
plt.xlabel("Variable")
plt.ylabel("Z ratio (anomaly mean / inlier mean)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
#Looking at the plot above:
#Variables Fixed Assets to Assets, 
# and Current Ratio have the exact same Zratio

#Variables Net Value Growth Rate, and Revenue per person have the exact same Zratio
#BUT... Revenue Per Share (Yuan ¥) is almost the exact same

# %%
identical_vars1 = df2[[' Fixed Assets to Assets', ' Current Ratio']].copy()
identical_vars1_corr = identical_vars1.corr()
# %%
identical_vars1.max()
#location of maximum value in the dataframe
identical_vars1.idxmax()
# %%
#Fixed Assets to Assets at 2344+1=2345
print(max(Xvar.iloc[2344, :]))
# %%
((Xvar.iloc[2344, :]).sort_values(ascending=False)).head(10)
#The abnormally high values 
# in row 2344 are here.
"""Fixed Assets to Assets             8.320000e+09
Total Asset Growth Rate            7.080000e+09
Allocation rate per person         3.480000e+09
Cash/Current Liability             2.090000e+09
Cash Turnover Rate                 1.300000e+09
Average Collection Days            1.370000e+08
Inventory Turnover Rate (times)    9.550000e+07"""

print(max(Xvar.iloc[2499, :]))
#value is 7130000000.0 which again makes no sense
#When trying to see if there any of other
#strange values in the billions
(Xvar.iloc[2499, :].sort_values(ascending=False)).head(10)
#The abnormally high values 
# in row 2499 are here.
"""Total Asset Growth Rate                7.130000e+09
Current Ratio                          2.750000e+09
Cash Turnover Rate                     1.590000e+09
Current Asset Turnover Rate            6.740000e+08
Quick Asset Turnover Rate              1.320000e+08
Inventory Turnover Rate (times)        1.910000e+07"""




#So if we put the two subsets together we see which
#Variables we have in common.

"""In common (both rows):

Total Asset Growth Rate
Cash Turnover Rate
Inventory Turnover Rate (times)


Only in row 2344:

Fixed Assets to Assets
Allocation rate per person
Cash/Current Liability
Average Collection Days


Only in row 2499:

Current Ratio
Current Asset Turnover Rate
Quick Asset Turnover Rate"""

# %%
#For those variables in common
#Let's do a matrix scatter plot to see if they are indeed almost identical
common_Xvar = Xvar[[' Total Asset Growth Rate', ' Cash Turnover Rate', ' Inventory Turnover Rate (times)']]
sns.pairplot(common_Xvar,corner=True, diag_kind='kde',
             plot_kws={'alpha':0.5, 'color': BLUE, 'edgecolor': 'none'}, diag_kws={'color': BLUE, 'fill': True})
plt.show()

# Cash Turnover Rate', ' Inventory Turnover Rate (times)
# are extremely skewed to the right
#Total asset growth rate is bimodal.

#Cash Turnover Rate and Inventory Turnover Rate (times) are almost identical, 
# but Cash Turnover decays slower.

#Overall it tells us with the 3 key variables that
# cause all 3 models to flag as outliers, that majority
# of the outliers are caused by extremely high values 
# in Total Asset Growth Rate, Cash Turnover Rate, 
# and Inventory Turnover Rate (times) which occur at a very small
# frequency.

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
#with the corrupted share worked out in each title, side by side in one figure
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, var in zip(axes, [' Total Asset Growth Rate', ' Cash Turnover Rate', ' Research and development expense rate']):
    sns.histplot(Xvar[var], bins=50, color=BLUE, edgecolor=SURFACE, linewidth=0.8, ax=ax)
    share = (Xvar[var] >= 1000).mean()
    ax.set_title(f"{var.strip()}\n{share:.0%} of entries are corrupted")
    ax.xaxis.set_major_formatter(BILLIONS)
    ax.set_xlabel("")
    ax.set_ylabel("")
axes[0].set_ylabel("Number of companies")
fig.supxlabel("Reported value (genuine values lie between 0 and 1, inside the first bar)", fontsize=10, color=INK_2)
fig.suptitle("Three variables with over 50% corrupted entries", x=0.01, ha='left', fontsize=14, fontweight='semibold')
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
Xvarqt.to_csv(HERE / 'X_Clean.csv')
Yvar.to_csv(HERE / 'Y_Clean.csv')
# %%
#============================================================================================================================================================================================================================================================================================================================================================================
#RUBEN'S t-SNE

Xvarqt_array = np.array(Xvarqt)
from sklearn.manifold import TSNE
perp=100 #perplexity value, can be changed to see how it affects the results
X_reduced = TSNE(n_components=2, perplexity=perp, 
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

# %%
from sklearn.cluster import DBSCAN

y  = np.ravel(Yvar)
c1, c2 = X_reduced[:, 0], X_reduced[:, 1]
clusters = DBSCAN(eps=3, min_samples=10).fit_predict(X_reduced)
clusters = np.where(clusters == -1, -1, clusters + 1) #Number clusters from 1; noise stays -1
g = sns.JointGrid(height=9)

# Contours: one colour per cluster (non-bankrupt only, skipping the DBSCAN noise label -1)
mask = (y == 0) & (clusters != -1)
sns.kdeplot(x=c1[mask], y=c2[mask], hue=clusters[mask], palette='Set2',
            levels=6, fill=True, alpha=0.5, legend=False, ax=g.ax_joint,thresh=0.075)

# Bankrupt points on top
g.ax_joint.scatter(c1[y == 1], c2[y == 1], s=60, color=RED,
                   edgecolors='white', linewidths=0.6, zorder=3, label='Bankrupt')

# Histogram along each axis, split by class
pal = {0: BLUE, 1: RED}
sns.histplot(x=c1, hue=y, palette=pal, bins=50, stat='density', common_norm=False,
             element='step', legend=False, ax=g.ax_marg_x)
sns.histplot(y=c2, hue=y, palette=pal, bins=50, stat='density', common_norm=False,
             element='step', legend=False, ax=g.ax_marg_y)

g.set_axis_labels('t-SNE Component 1', 't-SNE Component 2', fontsize=14)
g.ax_joint.legend(loc='upper right')
g.figure.suptitle(f't-SNE Visualization of Bankruptcy Data, perplexity = {perp}', fontsize=16, y=1.02)

#Cluster number at the centre of each cluster (noise -1 skipped)
for cl in np.unique(clusters[clusters != -1]):
    cx, cy = np.median(c1[clusters == cl]), np.median(c2[clusters == cl])
    g.ax_joint.text(cx, cy, str(cl), ha='center', va='center',
                    fontsize=9, weight='bold', color='black',
                    bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='none', alpha=0.8))


plt.show()
# %%
#%% (WARNING: this cell takes a long time to run, approx. 10 minutes on a laptop)
#KL divergence and trustworthiness across perplexities
from sklearn.manifold import trustworthiness

perplexities = [5, 15, 30, 50, 100, 150, 200, 300]
rows = []
for p in perplexities:
    tsne = TSNE(n_components=2, perplexity=p, early_exaggeration=12.0, learning_rate='auto',
                max_iter=1000, init='random', random_state=45735476,
                method='barnes_hut', angle=0.8, n_jobs=-1)
    emb = tsne.fit_transform(Xvarqt_array)
    rows.append({'Perplexity': p,
                 'KL': tsne.kl_divergence_,
                 'T_5': trustworthiness(Xvarqt_array, emb, n_neighbors=5),
                 'T_15': trustworthiness(Xvarqt_array, emb, n_neighbors=15),
                 'T_50': trustworthiness(Xvarqt_array, emb, n_neighbors=50)})
    print(f"perplexity {p} done")

tsne_scores = pd.DataFrame(rows)
print(tsne_scores.round(4))

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(tsne_scores['Perplexity'], tsne_scores['KL'], marker='o', color=BLUE)
ax1.set_title('KL divergence (lower = better fit)')
ax1.set_xlabel('Perplexity')
ax1.set_ylabel('KL divergence')

for col, c in zip(['T_5', 'T_15', 'T_50'], [BLUE, YELLOW, RED]):
    ax2.plot(tsne_scores['Perplexity'], tsne_scores[col], marker='o', color=c,
             label=f"k = {col.split('_')[1]}")
ax2.axvline(perp, color=MUTED, ls='--', lw=1)
ax2.set_title('Trustworthiness (higher = better)')
ax2.set_xlabel('Perplexity')
ax2.set_ylabel('Trustworthiness')
ax2.legend()

fig.suptitle('t-SNE quality across perplexity values', fontsize=15, x=0.01, ha='left')
fig.tight_layout()
plt.show()
#%%
#Bankruptcy rate per DBSCAN cluster, excluding noise points (-1)
keep = clusters != -1
cluster_summary = (pd.DataFrame({'Cluster': clusters[keep], 'Bankrupt': y[keep]})
                   .groupby('Cluster')['Bankrupt']
                   .agg(Companies='size', Bankrupt='sum', Pct_bankrupt='mean'))
cluster_summary['Pct_bankrupt'] = (cluster_summary['Pct_bankrupt'] * 100).round(2)
cluster_summary['Share_of_all_bankrupt'] = (cluster_summary['Bankrupt'] / y.sum() * 100).round(1)
cluster_summary['Lift'] = (cluster_summary['Pct_bankrupt'] / (y.mean() * 100)).round(2)
print(cluster_summary.sort_values('Pct_bankrupt', ascending=False))


#FROM: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
#https://github.com/scikit-learn/scikit-learn/blob/afe5bcd80/sklearn/manifold/_t_sne.py
# %%

#==============================================================================================================================================================================================================================================================================================================================================================================
#MUHAMMAD'S PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.makedirs('Figures', exist_ok=True)
# %%
X = pd.read_csv('X_Clean.csv' , index_col = 0)
y = pd.read_csv('Y_Clean.csv')['Bankrupt?']
print(y.shape)
print(X.columns)
# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=64, random_state=45735476, svd_solver='full')
components = pca.fit_transform(X)

#Number of components to be retained using Kaiser's Criterion
#Use the mean eigenvalue
thres = sum(pca.explained_variance_)/64

n = 0
for i in range(pca.explained_variance_.shape[0]):
    if pca.explained_variance_[i] > thres:
        print (pca.explained_variance_[i])
        n = n + 1

print(n, 'components are to be retained.', 'with total variance explained of: ', sum(pca.explained_variance_ratio_[:n])*100, '%')

plt.figure(figsize=(10, 8))
plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Principal Components')
plt.plot(range(1,len(pca.explained_variance_ratio_)+1),
         np.cumsum(pca.explained_variance_ratio_),
         color='blue',
         label='Cumulative Explained Variance')
plt.axvline(x=n, color='r', label = 'Cutoff for variables retained')
plt.legend(loc='best')
plt.savefig('Figures/Cumulative Explained Variance.png', dpi=300)
plt.show()
# %%
#Here we create the full on biplot.
idx = np.argsort(y)
def biplot(score, coeff, labels=None):
    xs = score[:, 0]
    ys = score[:, 1]
    plt.figure(figsize=(10, 8))
    plt.scatter(xs, ys, c=y, cmap='coolwarm')
    plt.colorbar(label='Bankruptcies')

    lengths = np.sqrt(coeff[:, 0]**2 + coeff[:, 1]**2)
    top = np.argsort(lengths)[-3:]

    factor = max(np.abs(xs).max(), np.abs(ys).max()) / lengths.max()

    for j in top:

        plt.arrow(0, 0, coeff[j, 0]*factor, coeff[j, 1]*factor, color='w', alpha=0.5, head_width=0.1, head_length=0.1)
        plt.text(coeff[j, 0] * factor, coeff[j, 1] * factor, labels[j], color='w', horizontalalignment='left', verticalalignment='center')


    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Biplot")
    plt.savefig('Figures/PCA Biplot.png', dpi=300, bbox_inches='tight')
    plt.show()

biplot(components, pca.components_.T, X.columns)
# %%
#3D plot to see if that helps to better visualize the data
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(111, projection='3d')

scatter_3d =ax.scatter(components[idx, 0], components[idx, 1], components[idx, 2], c=y.iloc[idx], cmap='coolwarm')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.colorbar(scatter_3d, label='Bankruptcies', shrink=0.7)
plt.savefig('Figures/3D Exploratory Plot.png', dpi=300)
plt.show()
# %%
#Loadings to see which factors influence the most here.
for x in range(2):
    loading_scores = pd.Series(pca.components_[x], index = X.columns)
    sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)
    top_10_factors = sorted_loading_scores.index[:10].values
    print( 'Most influential factors for PC', x+1,'\n', loading_scores[top_10_factors] , '\n\n')

#=============================================================================================================================================================================================================================================================================================================================================================================

# %%
#EXTRA: 3D Plot of the t-SNE results, using the first 3 components instead of 2
# %%
X_reduced3 = TSNE(n_components=3, perplexity=perp,
     early_exaggeration=12.0,
     learning_rate='auto',
     max_iter=1000,
     n_iter_without_progress=300,
     min_grad_norm=1e-07,
     metric='euclidean',
     init='random',
     verbose=2,
     random_state=45735476,
     method='barnes_hut',
     angle=0.8,
     n_jobs=-1).fit_transform(Xvarqt_array)
# %%
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(projection='3d')

#Same Set2 colours as the 2D contours, so cluster 1 is the same colour in both plots
cluster_ids = np.unique(clusters[clusters != -1])
cluster_cols = dict(zip(cluster_ids, sns.color_palette('Set2', len(cluster_ids))))

#Non-bankrupt points, one colour per cluster
for cl in cluster_ids:
    pts = X_reduced3[(clusters == cl) & (y == 0)]
    ax.scatter(*pts.T, s=4, color=cluster_cols[cl], alpha=0.3, label=f'Cluster {cl}')

#Noise points (-1) in faint grey
noise = X_reduced3[(clusters == -1) & (y == 0)]
ax.scatter(*noise.T, s=3, color=MUTED, alpha=0.1, label='Noise')

#Bankrupt points on top
ax.scatter(*X_reduced3[y == 1].T, s=35, color=RED, edgecolors='white', linewidths=0.4,
           alpha=1, label='Bankrupt', depthshade=False)

#Cluster number at the centre of each cluster
for cl in cluster_ids:
    cx, cy, cz = np.median(X_reduced3[clusters == cl], axis=0)
    ax.text(cx, cy, cz, str(cl), fontsize=10, weight='bold', color='black', ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.2', fc='white', ec='none', alpha=0.8), zorder=10)

#Black panes and dim gridlines (3D axes ignore the facecolor in rcParams)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((0, 0, 0, 1))
    axis.pane.set_edgecolor(GRID)
    axis._axinfo['grid']['color'] = GRID

ax.set_xlabel('t-SNE Component 1', labelpad=8)
ax.set_ylabel('t-SNE Component 2', labelpad=8)
ax.set_zlabel('t-SNE Component 3', labelpad=8)
ax.view_init(elev=20, azim=45)
leg = ax.legend(loc='upper right', markerscale=3)
for h in leg.legend_handles: h.set_alpha(1) #Solid legend markers even though the points are faint
ax.set_title(f'3D t-SNE coloured by DBSCAN cluster, perplexity = {perp}')
plt.tight_layout()
plt.show()

# %%
