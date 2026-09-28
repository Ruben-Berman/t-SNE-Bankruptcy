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
import pandas as pd #for csv importation
import seaborn as sns #for heatmap, pairplot and scatter
import matplotlib.pyplot as plt #for label
import numpy as np #For numerical operations
#From the pyod library I discovered on a LinkedIn post
from pyod.models.iforest import IForest #Good outlier detection model
from pyod.models.ecod import ECOD #Also good, uses an ECDF
from pyod.models.copod import COPOD #Like ECDF but uses a copula
from sklearn.preprocessing import QuantileTransformer, StandardScaler #For scaling variables with extreme outliers
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

flag_cols = [c for c in Xvar_reduced.columns if c.endswith('(encode)')]
cont_cols = Xvar_reduced.columns.drop(flag_cols)
# %%
Xvar_scaled = Xvar_reduced.copy()

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
# %%
seed = 45735476
qt = QuantileTransformer(n_quantiles=100,random_state=seed)





#THIS IS THE DATA THAT WILL BE USED
#FOR THE T-SNE AND PCA
Xvarqt = pd.DataFrame(qt.fit_transform(Xvar_scaled),
                      columns=Xvar_scaled.columns,
                      index=Xvar_scaled.index)
# %%
#============================================================================================================================================================================================================================================================================================================================================================================
#RUBEN'S t-SNE

#==============================================================================================================================================================================================================================================================================================================================================================================
#MUHAMMAD'S PCA


#=============================================================================================================================================================================================================================================================================================================================================================================
