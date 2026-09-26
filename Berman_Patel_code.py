"""
#Hello this is being written by Ruben Berman 45735476 to test if this works.
#You can use this code to test if the code works.
#Right now Code Appendix is what is here
"""
# %%
#Data importation with pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
#Dropping variables that are not useful for 
# prediction, such as the target variable 
# "Bankrupt?" and the binary variable 
# " Net Income Flag" 
# Bankrupt? is the target variable, 
# and we do not want to use it as a 
# predictor. Net Income Flag is a 
# binary variable, and has no variation, 
# so it is 
# not useful.
Xvar = df2.drop([" Net Income Flag","Bankrupt?"], axis=1)

# Yvar is the target variable, 
# which is the variable we want to predict. 
# In this case, it is the "Bankrupt?" variable, 
# which is a binary variable indicating whether 
# a company is bankrupt or not.
Yvar = df2["Bankrupt?"]
# %%
#Original correlation matrix of the 
# predictors, to see if there are any 
# highly correlated variables. 
# This is important because 
# highly correlated variables 
# can cause problems.
corr_matrix = Xvar.corr()

#Let's visualise the correlation matrix 
# with a heatmap, 
# that is triangular, 
# since the correlation 
# atrix is symmetric, 
# and we don't need to see 
# the upper triangle.
import numpy as np
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='turbo')
plt.show()




# %%
#Going to vectorize the correlation matrix to 
# find the highest correlation between predictors
corr_pairs = corr_matrix.unstack()
# %%
#Now dropping the duplicates and sorting the 
# correlation pairs
sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)

# %%Let's remove essentially the same pairs, 
# since correlation is symmetric. 
# For example, if A and B have a 
# correlation of 0.8, then B and A 
# will also have a correlation of 0.8. 
# We only need to keep one of these pairs.
#To do this, we can create a set to keep 
# track of the pairs we have already seen, 
# and only add a pair to the sorted_pairs 
# if it is not already in the set.


# %%

#Because we do not want to lose 
# too much information, we will 
# only drop variables that have 
# a correlation of 0.95 or higher 
# with another variable. 
# This is a common threshold used in 
# practice, and it will help us to reduce 
# the number of variables without losing 
# too much information.
unique_pairs_filtered = unique_pairs[unique_pairs.abs() > 0.95]

#Now we can drop the variables that are highly 
# correlated with other variables, and keep only
# one of them. For example, if A and B have a 
# correlation of 0.9, we can drop B and keep A.
#We can do this by creating a list of variables 
# to drop, and then dropping them from the 
# dataset.
variables_to_drop = []
unique_pairs_filtered = unique_pairs_filtered.sort_values(ascending=False)
for pair in unique_pairs_filtered.index:
    var1, var2 = pair
    if var1 not in variables_to_drop and var2 not in variables_to_drop:
        variables_to_drop.append(var2)
# %%
#Now we can drop the variables from the dataset
Xvar_reduced = Xvar.drop(variables_to_drop, axis=1)

#New heatmap of the correlation matrix, 
# to see if we have removed the highly 
# correlated variables
corr_matrix2 = Xvar_reduced.corr()
mask2 = np.triu(np.ones_like(corr_matrix2, dtype=bool))
sns.heatmap(corr_matrix2, mask=mask2, annot=False, cmap='turbo')
plt.title("Correlation Matrix after dropping highly correlated variables")
plt.show()



"""EXTRA TIPS
git pull                                  # 1. start of session: get partner's changes
# ... work, Ctrl+S as you go ...
git add .                                 # 2. stage everything you changed
git commit -m "Add t-SNE plot of ROA features"   # 3. snapshot with a message
git push                                  # 4. upload to GitHub
"""



# %%
#The amount of dimensions in the original 
# dataset is what exactly? Let's check the 
# shape of the original dataset, and the 
# shape of the dataset after dropping the 
# highly correlated variables.
print("Original dataset shape:", Xvar.shape)
print("Dataset shape after dropping highly correlated variables:", Xvar_reduced.shape)

#Went from 94 predictor variables to 79 variables which is a reduction of 15 variables, which is a good reduction in dimensionality, and will help with the curse of dimensionality.

# %%
#Isolation Forest, ECOD and COPOD are all 
# anomaly detection algorithms that can be 
# used to detect outliers in the dataset. 
# We will use Isolation Forest for this task, 
# since it is a popular and effective algorithm 
# for anomaly detection.
from  pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD

#True positive amount is the amount 
# of bankrupt companies in the dataset, 
# and total amount is the total amount of 
# companies in the dataset. 
# The contamination parameter 
# is the proportion of outliers in the 
# dataset, which is the amount of bankrupt 
# companies divided by the total amount of 
# companies.
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
# %%
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
        "Total rows flagged as anomalies by all 3 models",
        "Flagged rows that are actual bankruptcies (check for what's driving them)",
        "Flagged rows that are NOT bankruptcies (check for mis-imputation)",
    ],
}, index=["Total anomalies", "Positive (bankrupt)", "Negative (not bankrupt)"])
print(anomaly_summary.to_string())
# %%

# %%
mu = Xvar_reduced.mean()
s = Xvar_reduced.std()
#The absolute Z score, because it is
#A question of now whether the movement
#in centrality is caused by an extremely 
# large outlier or extremely small outlier
#But whether there is an outlier at all
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
plt.figure(figsize=(12,6))
plt.bar(Zlistfinal_df['Variable'], Zlistfinal_df['Zratio'])
plt.title("Z ratio of anomaly vs inlier group means")
plt.xlabel("Variable")
plt.ylabel("Z ratio (anomaly mean / inlier mean)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
#variables 63, 20 have the exact same Zratio
#variables 63, 23 have the exact same Zratio
#variables 20, 38 have the exact same Zratio

#Will continue tomorrow.