"""
#Hello this is being written by Ruben Berman 45735476 to test if this works.
#You can use this code to test if the code works.
#Right now Code Appendix is what is here
"""
# %%
#Data importation with pandas
import pandas as pd
df = pd.read_csv('data.csv')
# %%
"""
#Some instructions on Git operations
git pull                                   # get partner's changes
git add .                                  # stage all your changes
git commit -m "describe what you changed"
git push                                   # upload to GitHub
"""
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(), annot=False, cmap='plasma')
plt.show()
# %%
#Note: Several correlation in predictors
df2 = df.copy()
df2 = df2.drop(" Net Income Flag", axis=1)
#Note: Net Income Flag is a binary variable, and has a correlation of NaN with all other variables, so it is not useful for prediction.
# %%
corr_matrix = df2.corr()
# %%
#Going to vectorize the correlation matrix to find the highest correlation between predictors
corr_pairs = corr_matrix.unstack()
# %%
#Now dropping the duplicates and sorting the correlation pairs
sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
#Now dropping all the pairs that are 1, since those are the same variable correlated with itself
sorted_pairs = sorted_pairs[sorted_pairs != 1]
# %%
#Let's make a plot of the correlation pairs to visualize the distribution of correlations between predictors
plt.figure(figsize=(10, 6))
sns.histplot(sorted_pairs, bins=30, kde=True)
plt.title('Distribution of Correlation Pairs')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.show()

# %%
