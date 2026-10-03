[33mcommit 05cb661122ef0e1ae620a7d4def469b6ee6c18c2[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m, [m[1;31morigin/main[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Merge: 52c75fc 0796696
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Sat Oct 3 14:26:28 2026 +0200

    Merge branch 'main' of https://github.com/Ruben-Berman/t-SNE-Bankruptcy
    
    # Conflicts:
    #       Berman_Patel_code.py

[33mcommit 52c75fc35083062020e2b6e07aed56bd62ea2e95[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Sat Oct 3 14:19:50 2026 +0200

    fixed t-SNE useing perplexity 150

[33mcommit 079669694ef375dfcc7e993920c19c2258c3fac9[m
Author: MuhammadP17 <basef973@gmail.com>
Date:   Sat Oct 3 11:52:40 2026 +0200

    More changes, small things.

[33mcommit 3fbf7a28bdb9e932bbc7bf84b472415b4ea8a6d9[m
Author: MuhammadP17 <basef973@gmail.com>
Date:   Sat Oct 3 10:39:13 2026 +0200

    Changed preprocessing to exclude quantile transformation on binary variables. Resulting in new X_clean.csv to be used. Small changes correspondingly as such to PCA too. Run your t-SNE with the new data, should all be on the repo.

[33mcommit 1f02f80f4ea5961ac919646dd0772d4c7fca3098[m
Author: MuhammadP17 <basef973@gmail.com>
Date:   Sat Oct 3 09:58:09 2026 +0200

    One line changed.

[33mcommit 28682a870f6101ec59808a41c1cd6f7b27ecb4ac[m
Merge: a63933c 79367a2
Author: MuhammadP17 <basef973@gmail.com>
Date:   Sat Oct 3 09:56:10 2026 +0200

    Merge remote-tracking branch 'origin/main'

[33mcommit a63933c62703b461712a6b8416e3cdd7c6450cea[m
Author: MuhammadP17 <basef973@gmail.com>
Date:   Sat Oct 3 09:53:38 2026 +0200

    More PCA math added.

[33mcommit 79367a2ef76d305ca8e9cd72bfdfae8c4adf338f[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Sat Oct 3 00:21:39 2026 +0200

    Added plot, will address later

[33mcommit f5dba53efd5cd4c6b9081de1bd563b58b3cf1e43[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Sat Oct 3 00:10:51 2026 +0200

    Tried making my own t-SNE is proven difficult, instead will most likely use the one from sklearn

[33mcommit e87583a152d583dbf036ab397d10e139466c54dc[m
Merge: 4567a02 1c2b488
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Fri Oct 2 12:59:47 2026 +0200

    Merge branch 'main' of https://github.com/Ruben-Berman/t-SNE-Bankruptcy

[33mcommit 4567a023d4b5e6042e996346a965ef2ae2055539[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Fri Oct 2 12:59:38 2026 +0200

    I am simply revisiting the script, and continuing to do t-SNE

[33mcommit 1c2b488ac8fb527842b4037f1eb83fa5d67d4e04[m
Author: MuhammadP17 <basef973@gmail.com>
Date:   Fri Oct 2 11:15:13 2026 +0200

    Full PCA script added, conditional to future changes if needed.

[33mcommit ff999892fb03725f6954066c932d959fe79aec45[m
Author: MuhammadP17 <basef973@gmail.com>
Date:   Tue Sep 29 10:14:41 2026 +0200

    Added export files.

[33mcommit 1b5ec7cd1898328f8eda9d033f70205fba9b712e[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Mon Sep 28 17:07:36 2026 +0200

    described the data that will be used indicating it clearly

[33mcommit a45b0f61d1bbd6684d32577b2c124c16b7c9299b[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Mon Sep 28 17:06:20 2026 +0200

    Added coding blocks for the t-SNE (Ruben) and PCA (Muhammad)

[33mcommit 5fd6c423ec39e664411e79d3973a94ad246d16c6[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Mon Sep 28 17:03:24 2026 +0200

    encoded 3 variables with the most error values, then applied KNN to the ones with the intermediate amount, then winsorised those with only a small few error values, then applied quantile transformation.

[33mcommit 4397589e03090d22094066817483166fa1453235[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Sun Sep 27 23:40:57 2026 +0200

    Traced Zratio anomalies to source rows, treated skewed variables, clustered Quick Asset Turnover Rate
    
    - Used duplicate Zratios to find which rows were inflating anomaly Z-scores (rows 2344, 2499)
    - Identified 3 variables in common driving the inflation: Total Asset Growth Rate, Cash Turnover Rate, Inventory Turnover Rate (times)
    - Matrix scatterplot of the three to inspect skew and structure directly
    - Winsorised Cash/Current Liability; quantile-transformed Total Asset Growth Rate toward a Gaussian shape
    - Cash Turnover Rate's scatter showed clustering; checked correlated variables and found Quick Asset Turnover Rate
    - K-means (k=2) split Quick Asset Turnover Rate into two clusters (possibly small vs. large firms); binary-encoded as a new feature
    - Remaining variables still need standard scaling before dimensionality reduction

[33mcommit 6e00e99f8c706a64a1d5ecd348c70a299cd72b8d[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Sun Sep 27 12:22:49 2026 +0200

    fixed errors in reduced X variables now, it filters correctly

[33mcommit b47a6796b755e73103dfe54410ea27124e18cf3f[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Sat Sep 26 23:01:57 2026 +0200

    Will continue tomorrow

[33mcommit 6cf9ef06b0843d6e99f1f5a1253229184002379d[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Sat Sep 26 22:32:17 2026 +0200

    Expanded on EDA, developed unsupervised outlier detection models

[33mcommit 12226b67accaaaeac6d68c3069d7f39eac8d9104[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Sat Sep 26 14:42:11 2026 +0200

    Removing the same pairs in the vectorised correlation matrix, having a vector of only unique coorelation pairs

[33mcommit 38fba76c00d01a709dd2d2c0364cf5276e9529c4[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Fri Sep 25 23:53:27 2026 +0200

    vectorised the correlation matrix, removed duplicates and 1's and made a plot to see the distibution of correlations, where they mostly sit

[33mcommit 201820c9bd189796e22dfd26951e5cb075494619[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Fri Sep 25 23:18:00 2026 +0200

    added pandas csv importation and git terminal instructions

[33mcommit 83a6ec2f04d4f60c3c5549f76b9faa0e7dc26540[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Thu Sep 24 16:18:37 2026 +0200

    Add analysis code and bankruptcy dataset

[33mcommit 6c78906b4dd87d713a747ab39ee136a137ab4a04[m
Author: Ruben-Berman <129544963+Ruben-Berman@users.noreply.github.com>
Date:   Thu Sep 24 15:42:23 2026 +0200

    Initial commit
