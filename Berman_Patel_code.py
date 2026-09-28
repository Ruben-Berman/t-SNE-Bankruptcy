#START OF CODE APPENDIX MADE BY RUBEN BERMAN 45735476============================================================================================================================================================
"""EXTRA TIPS
git pull                                  # 1. start of session: get partner's changes
# ... work, Ctrl+S as you go ...
git add .                                 # 2. stage everything you changed
git commit -m "Add t-SNE plot of ROA features"   # 3. snapshot with a message
git push                                  # 4. upload to GitHub
"""
# %%
#All library imports
import pandas as pd #for csv importation
import seaborn as sns #for heatmap, pairplot and scatter
import matplotlib.pyplot as plt #for label
import numpy as np #For numerical operations
#From the pyod library I discovered on a LinkedIn post
from pyod.models.iforest import IForest #Good outlier detection model
from pyod.models.ecod import ECOD #Also good, uses an ECDF
from pyod.models.copod import COPOD #Like ECDF but uses a copula
from sklearn.preprocessing import QuantileTransformer #For scaling variables with extreme outliers
from sklearn.cluster import KMeans #For variables where distinct clusters were identified and binary encoding was done
# %%
#Data importation with pandas
df = pd.read_csv('data.csv')
# %%
"""
#Some instructions on Git operations
git pull                                   # get partner's changes
git add .                                  # stage all your changes
git commit -m "describe what you changed"
git push                                   # upload to GitHub
"""

# %%
#Note: Several correlation in predictors
#Copying the dataset
df2 = df.copy()
#Net Income Flag has nothing but 1's
print(df[" Net Income Flag"])
#Net Income Flag was removed, and Response variable Bankrupt
Xvar = df2.drop([" Net Income Flag","Bankrupt?"], axis=1)

# Yvar is the target variable, 
# which is the variable we want to predict. 
Yvar = df2["Bankrupt?"]

#NEW FINAL PLAN===================================================================================================================================================================================================================================================================
# %%
#Count parameters with billions
Thres = 1e9
Xbil = (Xvar>=1000).sum()
Xbil_sort = Xbil.sort_values(ascending=False)
Xbil_summary = (round(Xbil_sort/(len(Xvar)),4))
# %%
#For the variables that have values in the billions
#over 50%
#We will now plot first of all and see how they look
print(Xbil_summary[Xbil_summary>=0.5])

corr_matrix = Xvar.corr('spearman')
corr_matrix[' Total Asset Growth Rate']
corr_matrix[corr_matrix[' Cash Turnover Rate'] != 1] 
# %%
print((corr_matrix[corr_matrix[' Cash Turnover Rate'] != 1]).unstack())

# %%
sns.scatterplot(x=Xvar[' Cash Turnover Rate'],y=Xvar[' ROA(C) before interest and depreciation before interest'])
plt.show()

# %%
#Original correlation matrix of the 
# predictors, to see if there are any 
# highly correlated variables. 
# This is important because 
# highly correlated variables 
# can cause problems.


topvars1 = (corr_matrix[' Total Asset Growth Rate'].drop(' Total Asset Growth Rate').abs()).sort_values(ascending=False).head(3)

sns.pairplot(Xvar[[' Total Asset Growth Rate'] + list(topvars1.index)], corner=True, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.show()
# %%
topvars2 =  (corr_matrix[' Cash Turnover Rate'].drop(' Cash Turnover Rate')).sort_values(ascending=False).head(3)

sns.pairplot(Xvar[[' Cash Turnover Rate'] + list(topvars2.index)], corner=True, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.show()
# %%

sns.histplot(Xvar[' Total Asset Growth Rate'])
plt.show()

# %%
sns.histplot(Xvar[' Cash Turnover Rate'])
plt.show()
# %%
Xvar['TAGR (ecode)'] = (Xvar[' Total Asset Growth Rate']>1).astype(int)
Xvar['CTR (encode)'] = (Xvar[' Cash Turnover Rate']>1).astype(int)
Xvar['R&D (encode)'] = (Xvar[' Research and development expense rate']>1).astype(int)

# %%
Xvar2 = Xvar.copy()
Xvar2 = Xvar2.drop([' Cash Turnover Rate',' Total Asset Growth Rate',' Research and development expense rate'],axis=1)
# %%
Xvar3 = Xvar2.copy()

# %%
#These ones we will apply KNN
Xbil_summary[(Xbil_summary<0.5) & (Xbil_summary>0.15)]
# %%

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)

"""Research and development expense rate    0.4375
Inventory Turnover Rate (times)          0.3336
Quick Asset Turnover Rate                0.3162
Operating Expense Rate                   0.3017
Current Asset Turnover Rate              0.1607
Fixed Assets Turnover Frequency          0.1565"""

# %%
intermediate_vars = list(Xbil_summary[(Xbil_summary >= 0.015) & (Xbil_summary < 0.5)].index)

Xvar3[intermediate_vars] = Xvar3[intermediate_vars].mask(Xvar3[intermediate_vars] > 1)
corr_nan = Xvar3.corr('spearman')
clean_vars = Xvar3.columns[(Xvar3 <= 1).all()]

for var in intermediate_vars:
    topvars = corr_nan.loc[clean_vars, var].abs().sort_values(ascending=False).head(4)
    neighbour_vars = list(topvars.index)
    print(var, '<-', neighbour_vars)
    filled = imputer.fit_transform(Xvar3[[var] + neighbour_vars])
    Xvar3[var] = filled[:, 0]
# %%
#Now for the rest, we will apply winsorisation on
lower_vars = list(Xbil_summary[(Xbil_summary < 0.015) & (Xbil_summary > 0)].index)
# %%
for var in lower_vars:
    clean = Xvar3[var][Xvar3[var] <= 1]
    Xvar3[var] = Xvar3[var].clip(upper=clean.quantile(0.99))
# %%
Xvar4 = Xvar3.copy()
# %%

corr_matrix = Xvar4.corr('spearman')

#Let's visualise the correlation matrix 
# with a heatmap, 
# that is triangular, 
# since the correlation 
# Matrix is symmetric, 
# it is redundant to see
# the upper triangle.
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='turbo')
plt.show()

#Therefore because of the asymmetric values in
#Total Asset Growth Rate & Cash Turnover Rate
#We can apply K means clustering and encode them


# %%
#Because we do not want to lose 
# too much information, we will 
# only drop variables that have 
# a correlation of 0.95 or higher 
# with another variable. 

unique_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)).stack()
unique_pairs_filtered = unique_pairs[unique_pairs.abs() > 0.95]

#Now we can drop the variables that are highly 
# correlated with other variables, and keep only
# one of them. For example, if A and B have a 
# correlation of 0.9, we can drop B and keep A.
#We can do this by creating a list of variables 
# to drop, and then dropping them from the 
# dataset.
# %%
variables_to_drop = []
unique_pairs_filtered = unique_pairs_filtered.sort_values(ascending=False)
for pair in unique_pairs_filtered.index:
    var1, var2 = pair
    if var1 not in variables_to_drop and var2 not in variables_to_drop:
        variables_to_drop.append(var2)
# %%
print(pd.DataFrame(variables_to_drop))
#29/94 variables dropped
# %%
# = 65 variables now
Xvar_reduced = Xvar4.drop(variables_to_drop, axis=1)
# %%
#New heatmap of the correlation matrix, 
# to see if we have removed the highly 
# correlated variables

corr_matrix2 = Xvar_reduced.corr()

mask2 = np.triu(np.ones_like(corr_matrix2, dtype=bool))
sns.heatmap(corr_matrix2, mask=mask2, annot=False, cmap='turbo')
plt.title("Correlation Matrix after dropping highly correlated variables")
plt.show()

#Overall very good, some correlations though entailing
#Duplicate variables of different official names,
#Which is can investigate the following way:


#Isolation Forest, ECOD and COPOD are all 
# anomaly detection algorithms that can be 
# used to detect outliers in the dataset. 
# The contamination parameter 
# is the proportion of outliers in the 
# dataset, which is the amount of bankrupt 
# companies divided by the total amount of 
# companies.
# %%
TPAMT = Yvar[df["Bankrupt?"]==1].shape[0]
TOTALAMT = Yvar.shape[0]
contam = TPAMT / TOTALAMT
print("Contamination (proportion of outliers):", f"{contam:.2%}")
clf1 = IForest(n_estimators=100, 
                       contamination=contam, 
                       random_state=45735476)

#ECOD class for Unsupervised Outlier Detection 
# Using Empirical Cumulative Distribution 
# Functions (ECOD) ECOD is a parameter-free, 
# highly interpretable outlier detection 
# algorithm based on empirical CDF functions. 
clf2 = ECOD(contamination=contam)

#COPOD class for Copula Based Outlier Detector. 
# COPOD is a parameter-free, highly interpretable
#  outlier detection algorithm based on empirical
#  copula models
clf3 = COPOD(contamination=contam)

#Fitting all the models
clf1.fit(Xvar_reduced)
clf2.fit(Xvar_reduced)
clf3.fit(Xvar_reduced)
# %%

#Getting the predicted labels from each model
#1 means an outlier, and 0 means an inlier.
Ypred1 = clf1.labels_
Ypred2 = clf2.labels_
Ypred3 = clf3.labels_

#Now we can combine the predictions from the 
# three models, and only keep the companies 
# that are predicted to be outliers by all 
# three models. This will help us to reduce the 
# number of false positives, and only keep the 
# companies that are most likely to be bankrupt.
anomalies = (
    (Ypred1==1) &
    (Ypred2==1) &
    (Ypred3==1)
)


Xanon = Xvar_reduced[anomalies]
Yanon = Yvar[anomalies]
# %%
n_anon = len(Yanon)
n_pos = int(np.sum(Yanon))
n_neg = n_anon - n_pos
anomaly_summary = pd.DataFrame({
    "Count": [n_anon, n_pos, n_neg],
    "Meaning": [
        "Total rows flagged by all 3 models",
        "Flagged rows that are actual bankruptcies (check for what's driving them)",
        "Flagged rows that are NOT bankruptcies (check for mis-imputation)",
    ],
}, index=["Total anomalies", "Positive (bankrupt)", "Negative (not bankrupt)"])
print(anomaly_summary.to_string())


# %%
mu = Xvar_reduced.mean()
s = Xvar_reduced.std()
#The absolute Z score, because it is
#A question of now whether the movement
#in centrality is caused by an extremely 
# large outlier or extremely small outlier
#But whether there is an outlier at all
# |Z| =|(X-mu)/sigma|
Zscoreord = np.abs((Xvar_reduced - mu)/s)
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
plt.bar(Zlistfinal_df['Variable'], Zlistfinal_df['Zratio'])
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

identical_vars1.max()
#location of maximum value in the dataframe
identical_vars1.idxmax()

#Fixed Assets to Assets at 2344+1=2345
max(Xvar_reduced.iloc[2344, :])

((Xvar_reduced.iloc[2344, :]).sort_values(ascending=False)).head(10)
#The abnormally high values 
# in row 2344 are here.
"""Fixed Assets to Assets             8.320000e+09
Total Asset Growth Rate            7.080000e+09
Allocation rate per person         3.480000e+09
Cash/Current Liability             2.090000e+09
Cash Turnover Rate                 1.300000e+09
Average Collection Days            1.370000e+08
Inventory Turnover Rate (times)    9.550000e+07"""

max(Xvar_reduced.iloc[2499, :])
#value is 7130000000.0 which again makes no sense
#When trying to see if there any of other
#strange values in the billions
(Xvar_reduced.iloc[2499, :].sort_values(ascending=False)).head(10)
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
common_Xvar = Xvar_reduced[[' Total Asset Growth Rate', ' Cash Turnover Rate', ' Inventory Turnover Rate (times)']]
sns.pairplot(common_Xvar,corner=True, diag_kind='kde', plot_kws={'alpha':0.5})
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

#But there is still an issue an 
# Asset growth rate of 9 billion makes no sense
# %%
common_Xvar[' Total Asset Growth Rate'].sort_values(ascending=False).head()
#
len(common_Xvar[common_Xvar[' Total Asset Growth Rate']>1])
#6017 entries out of the 6819 entries have a 
# Total Asset Growth Rate greater than 1, 
# which is 88.2% of the dataset. 
# This is a huge problem, and 
# indicates that there is a serious i
# ssue with the data quality.
# %%
len(common_Xvar[common_Xvar[' Cash Turnover Rate']>1])

#4239 entries out of 6819 entries have a 
# Cash Turnover rate of more than 1.
# But Cash Turnover rate is not a percentage
# So it is pretty normal for such massive values
# To exist here since it is potentially massive banks
#It is the Taiwanese Stock Exhange listed companies

#It is 2 clusters/groups
#Multi-national Firms
#Small Medium Enterprises/ domestic corporates

#But the majority based on the sea born plot are smaller
#entities


#And for the Inventory Turnover Rate
#Logically it is the same underlying principle of
#Cash turnover rate, but just in the context of physcially held
#goods/services
# %%


#For Cash/Current Liability  APPLIEDa slightly different story is told
# 3rd quartile is a percentage
#We can see again how many values are astronimically high
len(Xvar_reduced[Xvar_reduced[" Cash/Current Liability"]>1])

#Only 46 values above 100% this can be winsorised
#So about 0.006% are what will be changed, fixing
#dramatic skewness
#Let's winsorise now



# %%
CCL = Xvar_reduced[" Cash/Current Liability"]
CCL[CCL>0.50].sort_values(ascending=False)
# %%
#We see a progression from 74.5% then 1.00000 (100%) to amounts
#In the billions


#NOW WE BEGIN TREATMENT PHASES------------------
CCLMAX = 1
CCL_winsorized = CCL.clip(upper=CCLMAX)
# %%
Xvar_reduced_win = Xvar_reduced.copy()
Xvar_reduced_win = Xvar_reduced_win.drop([' Cash/Current Liability'], axis=1)


Xvar_reduced_win['Cash/Current Liability (winsorised)'] = CCL_winsorized


# %%
common_Xvar[' T'].sort_values(ascending=False).head()
TAGR = common_Xvar[' Total Asset Grootal Asset Growth Ratewth Rate']

qt = QuantileTransformer(n_quantiles=100,random_state=45735476)
TAGRqt = pd.DataFrame(qt.fit_transform(TAGR.to_frame()),
                      columns=["TAGR (QT)"],
                      index=TAGR.index)


Xvar_win_qt = Xvar_reduced_win.copy()
Xvar_win_qt['TAGR (Quantile Transformed)'] = TAGRqt
#END OF CODE APPENDIX MADE BY RUBEN BERMAN 45735476============================================================================================================================================================



#Due to the Clustering seen in the 
#Cash Turnover rate
# %%
(corr_matrix2[' Cash Turnover Rate'].drop([' Cash Turnover Rate'],axis=0)).sort_values(ascending=False)
# %%
#Plot now Quick Asset Turnover and Cash Turnover Rate
plt.scatter(y=Xvar_win_qt[' Cash Turnover Rate'],x=Xvar_win_qt[' Quick Asset Turnover Rate'])
plt.xlabel("Quick Asset Turnover Rate")
plt.ylabel("Cash Turnover Rate")
plt.show()
# %%

kmeans2 = KMeans(n_clusters=2, random_state=3, n_init=20).fit(Xvar_win_qt[[' Cash Turnover Rate',' Quick Asset Turnover Rate']])
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y=Xvar_win_qt[' Cash Turnover Rate'], x=Xvar_win_qt[' Quick Asset Turnover Rate'], c=kmeans2.labels_)
ax.set_title("K-Means Clustering Results with K=2")
plt.show()
# %%
Xvar_win_qt.drop([' Quick Asset Turnover Rate'],axis=1)
Xvar_win_qt["Quick Asset Turnover Rate (encoded)"] = kmeans2.labels_





# %%
#Clearly the one cluster is problem matic (purple)
#there are an overwhelming amount of zeroes in 
#Quick asset ratio, so let's apply KNN
#for just these 2 variables (Quick and Cash Turnover)



# %%

# %%


# %%
