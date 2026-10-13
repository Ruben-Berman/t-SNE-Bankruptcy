# t-SNE Bankruptcy

Dimensionality reduction and cluster analysis of the Taiwanese company bankruptcy dataset, using t-SNE and PCA.

## Academic Context

| Item | Detail |
|---|---|
| Module | STTN623 - Multivariate Statistics |
| Institution | North-West University (NWU) |
| Lecturer | Dr. Shawn Liebenberg |
| Authors | Ruben Berman (45735476), Muhammad Patel (45112894) |
| Submission date | 15/10/2026 |

## Dataset

- **Source:** Taiwan Economic Journal, 1999 to 2009
- **Size:** 6,819 companies, 95 financial ratios
- **Response:** `Bankrupt?` (1 = bankrupt, 0 = not bankrupt)
- **Predictors:** 94 financial ratios. `Net Income Flag` was removed because it equals 1 for every company.
- **Class imbalance:** 220 companies (3.23%) went bankrupt

## Methodology

### 1. Outlier Detection

- Three unsupervised models fitted on the raw data: Isolation Forest, ECOD and COPOD (via `pyod`)
- Contamination rate set equal to the bankruptcy rate (3.23%)
- Only rows flagged by all three models are treated as consensus anomalies
- Absolute Z-scores of the anomalies compared with the inliers to find which variables drive the flags. This is how the corrupted values below were found.

### 2. Corrupted Value Handling

Several ratio variables that should lie between 0 and 1 contain values in the billions. These are treated according to how many values are affected:

| Share of corrupted values | Treatment |
|---|---|
| Above 50% | Replaced with a binary flag (1 = corrupted, 0 = genuine) |
| 1.5% to 50% | KNN imputation (k = 5), using the 4 clean variables most correlated (Spearman) with each target |
| Below 1.5% | Winsorised at the 99th percentile of genuine values |

### 3. Feature Reduction and Scaling

- Spearman correlation matrix computed on the cleaned data
- 51 variable pairs have an absolute correlation above 0.95. They are processed greedily from the highest correlation down, and the second variable of a pair is dropped unless one of the two has already been dropped.
- Result: 30 variables dropped, 64 kept. Every dropped variable still has a near-duplicate (|ρ| > 0.95) among the 64 that remain, so no unique information is lost.
- Continuous variables transformed with a `QuantileTransformer` (normal output, 100 quantiles). Binary flags are left as they are.
- Final data exported as `X_Clean.csv` and `Y_Clean.csv`

### 4. t-SNE and Clustering

- 2D t-SNE (Barnes-Hut) with perplexity 100
- DBSCAN (`eps = 3`, `min_samples = 10`) applied to the 2D embedding, with clusters numbered from 1 and noise labelled -1
- Bankruptcy rate calculated per cluster, leaving out noise points
- Perplexity sweep over 5, 15, 30, 50, 100, 150, 200 and 300, comparing KL divergence and trustworthiness at k = 5, 15 and 50


### 5. PCA

- PCA fitted on the 64 cleaned variables
- Number of components kept using Kaiser's criterion (eigenvalue above the mean eigenvalue)
- Scree and cumulative explained-variance plot, a biplot, a 3D component plot, and loadings to find the most influential ratios

### 6. Visualisation

- Side-by-side 2D t-SNE and PCA maps, with density contours, bankrupt companies highlighted, and marginal histograms by class. DBSCAN clusters are shown on the t-SNE map only.
- Interactive 3D t-SNE (coloured by DBSCAN cluster) and 3D PCA (coloured by bankruptcy), opened in the browser with Plotly

## Key Results

- **Outlier detection:** each model flagged 220 companies, and all three agreed on 144 of them. 34 of those 144 went bankrupt (23.6%), about 7 times the overall rate. The consensus anomalies also led to the discovery of corrupted values in several ratio variables.
- **Perplexity:** `[RESULT: KL divergence and trustworthiness at perplexity 100 compared with the other values]`
- **t-SNE clusters:** DBSCAN found 5 clusters and 6 noise points. Bankruptcy is concentrated in two clusters:

  | Cluster | Companies | Bankruptcy rate |
  |---|---|---|
  | 1 | 1,652 | 9.50% |
  | 2 | 879 | 4.21% |
  | 3 | 2,422 | 0.87% |
  | 4 | 1,259 | 0.40% |
  | 5 | 601 | 0.00% |

- **Cluster 1 profile:** these are loss-making companies. Their tax rate is near zero, and EPS, ROA and retained earnings are all low; the approximate cut-offs (EPS ≤ 0.22, ROA(A) ≤ 0.54, retained earnings ≤ 0.93) appear to mark the break-even point. Unlike Cluster 2, Cluster 1 companies carry long-term debt (Long-term Liability to Current Assets, Cliff's δ = +1.00).
- **PCA:** Kaiser's criterion keeps 13 of the 64 components, which together explain 81.9% of the variance (PC1 23.2%, PC2 16.1%, PC3 9.4%). The first two components show one continuous cloud with no clear groups, which is why the clustering is done on t-SNE rather than PCA.

## Repository Structure

| File | Description |
|---|---|
| `Berman_Patel_code.py` | Full analysis pipeline, including the PCA section |
| `PCA.ipynb` | Original PCA notebook (Muhammad Patel) |
| `data.csv` | Raw bankruptcy dataset |
| `X_Clean.csv` | Cleaned and transformed predictors (64 variables) |
| `Y_Clean.csv` | Response variable |
| `Figures/` | Saved PCA figures |

## How to Run

1. Install the dependencies:

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn pyod plotly
```

2. Keep `data.csv` in the same folder as the script. The script finds its own folder, so no paths need changing.
3. Run `Berman_Patel_code.py` cell by cell (VS Code interactive window or Jupyter), or as a whole script.

**Notes:**

- Random seed: `45735476`
- The perplexity sweep cell takes about 10 minutes on a laptop
- The Plotly cells open the 3D plots in your default browser

## AI Use Declaration

AI assistance (Claude) was used for the following, and these sections are marked as such in the code:

- the 3D t-SNE plots
- the interactive Plotly visuals
- the side-by-side 2D t-SNE and PCA maps
- the dark plotting theme

## References

- Liang, D., Lu, C.-C., Tsai, C.-F. and Shih, G.-A. (2016). Financial ratios and corporate governance indicators in bankruptcy prediction: A comprehensive study. *European Journal of Operational Research*, 252(2), 561–572.
- van der Maaten, L. and Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
- Ester, M., Kriegel, H.-P., Sander, J. and Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of KDD-96*, 226–231.
- Zhao, Y., Nasrullah, Z. and Li, Z. (2019). PyOD: A Python toolbox for scalable outlier detection. *Journal of Machine Learning Research*, 20(96), 1–7.
- scikit-learn documentation: [`sklearn.manifold.TSNE`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
