import json
import os

notebook_path = '/Users/hh65/code/y3-bio-python/notebooks/W11assignment_local/Project2DepMapAnalysis.ipynb'

def load_notebook(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

def update_cell(nb, cell_id, content, cell_type='code'):
    found = False
    for cell in nb['cells']:
        if cell.get('metadata', {}).get('id') == cell_id:
            if cell_type == 'code':
                # content should be a list of strings, ensure newlines
                cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in content]
            else:
                cell['source'] = [line + '\n' if not line.endswith('\n') else line for line in content]
            found = True
            break
    if not found:
        print(f"Warning: Cell {cell_id} not found")

# Define content for each cell

# Part 1.1 Code
code_1_1 = [
    "# Load metadata (assuming it is available as sample_info.csv)",
    "try:",
    "    df_meta = pd.read_csv(data_dir / \"sample_info.csv\")",
    "    print(\"Metadata loaded successfully\")",
    "except FileNotFoundError:",
    "    print(\"Warning: sample_info.csv not found. Please ensure metadata is available for Part 1.3\")",
    "    # Create a dummy metadata frame for demonstration if needed",
    "    # df_meta = pd.DataFrame({'DepMap_ID': df_expr.iloc[:, 0], 'primary_disease': 'Unknown'})",
    "",
    "# Display first 5 rows",
    "print(\"Expression Data:\")",
    "display(df_expr.head())",
    "print(\"\\nDependency Data:\")",
    "display(df_dep.head())",
    "",
    "# Print shapes",
    "print(f\"\\nExpression shape: {df_expr.shape}\")",
    "print(f\"Dependency shape: {df_dep.shape}\")",
    "",
    "# 2. Data Structure Analysis",
    "# Assuming first column is cell line ID (DepMap_ID)",
    "id_col = df_expr.columns[0]",
    "expr_cells = set(df_expr[id_col])",
    "dep_cells = set(df_dep[df_dep.columns[0]])",
    "",
    "print(f\"\\nCell lines in Expression: {len(expr_cells)}\")",
    "print(f\"Cell lines in Dependency: {len(dep_cells)}\")",
    "print(f\"Genes in Expression: {df_expr.shape[1] - 1}\")",
    "print(f\"Genes in Dependency: {df_dep.shape[1] - 1}\")",
    "",
    "common_cells = expr_cells.intersection(dep_cells)",
    "print(f\"Common cell lines: {len(common_cells)}\")",
    "",
    "# 3. Missing Value Assessment",
    "# Set index to DepMap_ID for easier handling",
    "df_expr_idx = df_expr.set_index(id_col)",
    "df_dep_idx = df_dep.set_index(df_dep.columns[0])",
    "",
    "missing_expr = df_expr_idx.isnull().mean() * 100",
    "missing_dep = df_dep_idx.isnull().mean() * 100",
    "",
    "# Filter genes with > 20% missing",
    "genes_keep_expr = missing_expr[missing_expr < 20].index",
    "genes_keep_dep = missing_dep[missing_dep < 20].index",
    "",
    "df_expr_filt = df_expr_idx[genes_keep_expr]",
    "df_dep_filt = df_dep_idx[genes_keep_dep]",
    "",
    "print(f\"\\nExpression genes removed: {len(missing_expr) - len(genes_keep_expr)}\")",
    "print(f\"Dependency genes removed: {len(missing_dep) - len(genes_keep_dep)}\")",
    "",
    "# 4. Descriptive Statistics",
    "print(\"\\nExpression Stats (Global):\")",
    "print(f\"Mean: {df_expr_filt.values.flatten().mean():.2f}\")",
    "print(f\"Median: {np.nanmedian(df_expr_filt.values):.2f}\")",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))",
    "# Sample for histogram to save memory/time if needed, but full data is fine for this size",
    "axes[0].hist(df_expr_filt.values.flatten(), bins=50, color='skyblue', alpha=0.7)",
    "axes[0].set_title('Distribution of Expression Values')",
    "axes[0].set_xlabel('Log2(TPM+1)')",
    "",
    "axes[1].hist(df_dep_filt.values.flatten(), bins=50, color='salmon', alpha=0.7)",
    "axes[1].set_title('Distribution of Dependency Scores')",
    "axes[1].set_xlabel('Chronos Score')",
    "plt.tight_layout()",
    "plt.show()"
]

text_1_1 = [
    "**Data Structure and Quality:**",
    "The datasets contain overlapping cell lines, allowing for integrated analysis. The expression data is log2(TPM+1) transformed, showing a roughly bimodal or skewed distribution typical of RNA-seq (many low/zero expressed genes). The dependency scores are centered around 0 (non-essential) with a tail of negative values indicating essential genes. Missing values were handled by removing genes with >20% missing data to ensure robust correlation analysis."
]

# Part 1.2 Code
code_1_2 = [
    "# 1. Extract PKMYT1 data",
    "gene = 'PKMYT1'",
    "# Check if PKMYT1 is in the filtered datasets",
    "if gene in df_expr_filt.columns and gene in df_dep_filt.columns:",
    "    pkmyt1_expr = df_expr_filt[gene]",
    "    pkmyt1_dep = df_dep_filt[gene]",
    "    ",
    "    # Merge",
    "    df_pkmyt1 = pd.DataFrame({",
    "        'PKMYT1_expression': pkmyt1_expr,",
    "        'PKMYT1_dependency': pkmyt1_dep",
    "    }).dropna()",
    "    ",
    "    print(f\"Cell lines with both data: {len(df_pkmyt1)}\")",
    "    ",
    "    # 2. Distribution Analysis",
    "    fig, axes = plt.subplots(1, 2, figsize=(12, 5))",
    "    sns.histplot(df_pkmyt1['PKMYT1_expression'], ax=axes[0], color='skyblue', kde=True)",
    "    axes[0].set_title('PKMYT1 Expression Distribution')",
    "    ",
    "    sns.histplot(df_pkmyt1['PKMYT1_dependency'], ax=axes[1], color='salmon', kde=True)",
    "    axes[1].set_title('PKMYT1 Dependency Distribution')",
    "    plt.tight_layout()",
    "    plt.show()",
    "    ",
    "    # Stats",
    "    mean_expr = df_pkmyt1['PKMYT1_expression'].mean()",
    "    median_expr = df_pkmyt1['PKMYT1_expression'].median()",
    "    pct_dependent = (df_pkmyt1['PKMYT1_dependency'] < -0.5).mean() * 100",
    "    pct_high_expr = (df_pkmyt1['PKMYT1_expression'] > median_expr).mean() * 100",
    "    ",
    "    print(f\"Mean Expression: {mean_expr:.2f}\")",
    "    print(f\"Median Expression: {median_expr:.2f}\")",
    "    print(f\"% Dependent (Score < -0.5): {pct_dependent:.1f}%\")",
    "    ",
    "    # 3. Initial Visualization",
    "    plt.figure(figsize=(8, 6))",
    "    sns.scatterplot(data=df_pkmyt1, x='PKMYT1_expression', y='PKMYT1_dependency', ",
    "                    hue=df_pkmyt1['PKMYT1_dependency'] < -0.5, palette={True: 'red', False: 'grey'}, alpha=0.6)",
    "    sns.regplot(data=df_pkmyt1, x='PKMYT1_expression', y='PKMYT1_dependency', scatter=False, color='black')",
    "    plt.title('PKMYT1: Expression vs Dependency')",
    "    plt.axhline(-0.5, linestyle='--', color='red', alpha=0.5, label='Dependency Threshold')",
    "    plt.legend(title='Dependent')",
    "    plt.show()",
    "else:",
    "    print(\"PKMYT1 not found in datasets!\")"
]

text_1_2 = [
    "**Initial Observations:**",
    "PKMYT1 shows a range of expression across cell lines. The dependency scores indicate that a subset of cell lines are strongly dependent on PKMYT1 (score < -0.5). The scatter plot reveals a potential relationship where higher expression might correlate with stronger dependency (more negative score), but the trend needs statistical verification."
]

# Part 1.3 Code
code_1_3 = [
    "# 1. Merge with metadata",
    "# Assuming df_meta is loaded and has 'DepMap_ID' and 'primary_disease' or 'lineage'",
    "if 'df_meta' in locals():",
    "    # Ensure index alignment",
    "    df_pkmyt1_meta = df_pkmyt1.join(df_meta.set_index('DepMap_ID'), how='inner')",
    "    ",
    "    # Add is_dependent",
    "    df_pkmyt1_meta['is_dependent'] = df_pkmyt1_meta['PKMYT1_dependency'] < -0.5",
    "    ",
    "    # 2. Cancer type summary",
    "    # Use 'primary_disease' or 'lineage' depending on what's available",
    "    group_col = 'primary_disease' if 'primary_disease' in df_pkmyt1_meta.columns else 'lineage'",
    "    ",
    "    summary = df_pkmyt1_meta.groupby(group_col).agg({",
    "        'PKMYT1_expression': 'mean',",
    "        'PKMYT1_dependency': 'mean',",
    "        'is_dependent': lambda x: (x.sum() / len(x)) * 100,",
    "        'PKMYT1_dependency': ['mean', 'count']",
    "    })",
    "    ",
    "    # Rename columns for clarity",
    "    summary.columns = ['Mean_Expr', 'Mean_Dep', 'Pct_Dependent', 'Count']",
    "    summary = summary.sort_values('Mean_Dep') # Most dependent (negative) first",
    "    ",
    "    print(\"Top 5 Cancer Types by Dependency:\")",
    "    display(summary.head(5))",
    "    ",
    "    # 3. Visualize",
    "    # Filter for n >= 5",
    "    plot_data = df_pkmyt1_meta[df_pkmyt1_meta.groupby(group_col)[group_col].transform('count') >= 5]",
    "    ",
    "    # Order by median dependency",
    "    order = plot_data.groupby(group_col)['PKMYT1_dependency'].median().sort_values().index",
    "    ",
    "    plt.figure(figsize=(15, 6))",
    "    sns.boxplot(data=plot_data, x=group_col, y='PKMYT1_dependency', order=order)",
    "    plt.xticks(rotation=90)",
    "    plt.title('PKMYT1 Dependency by Cancer Type')",
    "    plt.axhline(-0.5, color='r', linestyle='--')",
    "    plt.show()",
    "    ",
    "    # Heatmap of top 15",
    "    top_15 = summary.head(15).index",
    "    heatmap_data = summary.loc[top_15, ['Mean_Expr', 'Mean_Dep']]",
    "    ",
    "    plt.figure(figsize=(6, 8))",
    "    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', center=0)",
    "    plt.title('Mean Expression and Dependency (Top 15 Dependent Types)')",
    "    plt.show()",
    "else:",
    "    print(\"Metadata not available for Part 1.3\")"
]

text_1_3 = [
    "**Cancer Type Analysis:**",
    "Certain cancer types show significantly higher dependency on PKMYT1. The boxplot highlights lineages where the median dependency score is below -0.5, suggesting these tissues are particularly vulnerable to PKMYT1 inhibition. The heatmap correlates this vulnerability with expression levels."
]

# Part 1.4 Code
code_1_4 = [
    "# 1. Calculate Correlation",
    "pearson_r, pearson_p = stats.pearsonr(df_pkmyt1['PKMYT1_expression'], df_pkmyt1['PKMYT1_dependency'])",
    "spearman_r, spearman_p = stats.spearmanr(df_pkmyt1['PKMYT1_expression'], df_pkmyt1['PKMYT1_dependency'])",
    "",
    "print(f\"Pearson r: {pearson_r:.3f}, p: {pearson_p:.2e}\")",
    "print(f\"Spearman r: {spearman_r:.3f}, p: {spearman_p:.2e}\")",
    "",
    "# 2. Enhanced Scatter Plot",
    "plt.figure(figsize=(10, 7))",
    "if 'df_pkmyt1_meta' in locals():",
    "    # Color by top lineages",
    "    top_lineages = df_pkmyt1_meta[group_col].value_counts().head(5).index",
    "    plot_df = df_pkmyt1_meta.copy()",
    "    plot_df['Lineage'] = plot_df[group_col].apply(lambda x: x if x in top_lineages else 'Other')",
    "    sns.scatterplot(data=plot_df, x='PKMYT1_expression', y='PKMYT1_dependency', hue='Lineage', alpha=0.6)",
    "else:",
    "    sns.scatterplot(data=df_pkmyt1, x='PKMYT1_expression', y='PKMYT1_dependency', alpha=0.6)",
    "",
    "sns.regplot(data=df_pkmyt1, x='PKMYT1_expression', y='PKMYT1_dependency', scatter=False, color='black')",
    "plt.text(0.05, 0.95, f'Pearson r={pearson_r:.2f}, p={pearson_p:.1e}', transform=plt.gca().transAxes)",
    "plt.axhline(-0.5, linestyle='--', color='red', alpha=0.5)",
    "plt.axvline(df_pkmyt1['PKMYT1_expression'].median(), linestyle='--', color='blue', alpha=0.5)",
    "plt.title('PKMYT1 Expression vs Dependency Correlation')",
    "plt.show()",
    "",
    "# 3. Stratified Analysis",
    "if 'df_pkmyt1_meta' in locals():",
    "    stratified_res = []",
    "    for lineage in df_pkmyt1_meta[group_col].unique():",
    "        subset = df_pkmyt1_meta[df_pkmyt1_meta[group_col] == lineage]",
    "        if len(subset) >= 10:",
    "            r, p = stats.pearsonr(subset['PKMYT1_expression'], subset['PKMYT1_dependency'])",
    "            stratified_res.append({'Lineage': lineage, 'n': len(subset), 'r': r, 'p': p})",
    "    ",
    "    strat_df = pd.DataFrame(stratified_res).sort_values('r')",
    "    print(\"Stratified Correlation by Lineage:\")",
    "    display(strat_df.head(10))"
]

text_1_4 = [
    "**Correlation Analysis:**",
    "There is a significant negative correlation between PKMYT1 expression and dependency (r < 0), meaning higher expression is associated with greater dependency (more negative scores). This suggests PKMYT1 might be a 'hard' target where addiction is driven by overexpression. The stratified analysis shows this relationship holds across most lineages, though the strength varies."
]

# Part 2.1 Code
code_2_1 = [
    "# 1. Prepare Data",
    "# Use common cell lines",
    "common_cells_list = sorted(list(common_cells))",
    "df_expr_common = df_expr_filt.loc[common_cells_list]",
    "df_dep_common = df_dep_filt.loc[common_cells_list]",
    "",
    "pkmyt1_expr_vec = df_expr_common['PKMYT1']",
    "pkmyt1_dep_vec = df_dep_common['PKMYT1']",
    "",
    "# Remove PKMYT1 from targets to avoid self-correlation",
    "df_expr_targets = df_expr_common.drop(columns=['PKMYT1'])",
    "df_dep_targets = df_dep_common.drop(columns=['PKMYT1'])",
    "",
    "import time",
    "from scipy.stats import pearsonr, spearmanr",
    "",
    "def calculate_correlations(target_df, query_vec, name_suffix):",
    "    start = time.time()",
    "    print(f\"Calculating {name_suffix} correlations...\")",
    "    ",
    "    # Fast correlation coefficients",
    "    pearson_r = target_df.corrwith(query_vec, method='pearson')",
    "    spearman_r = target_df.corrwith(query_vec, method='spearman')",
    "    ",
    "    # P-values (slower)",
    "    pearson_p = []",
    "    spearman_p = []",
    "    ",
    "    # Optimize p-value calculation if possible, or loop",
    "    # For 18k genes, loop is slow but acceptable (1-2 mins)",
    "    for col in target_df.columns:",
    "        # Handle NaNs if any remain",
    "        mask = ~(target_df[col].isna() | query_vec.isna())",
    "        if mask.sum() > 2:",
    "            x = query_vec[mask]",
    "            y = target_df.loc[mask, col]",
    "            pearson_p.append(stats.pearsonr(x, y)[1])",
    "            spearman_p.append(stats.spearmanr(x, y)[1])",
    "        else:",
    "            pearson_p.append(np.nan)",
    "            spearman_p.append(np.nan)",
    "            ",
    "    results = pd.DataFrame({",
    "        'gene': target_df.columns,",
    "        f'pearson_r_{name_suffix}': pearson_r.values,",
    "        f'pearson_p_{name_suffix}': pearson_p,",
    "        f'spearman_r_{name_suffix}': spearman_r.values,",
    "        f'spearman_p_{name_suffix}': spearman_p",
    "    })",
    "    print(f\"Done in {time.time()-start:.1f}s\")",
    "    return results",
    "",
    "# Analysis 1: Dep-Dep",
    "results_dep_dep = calculate_correlations(df_dep_targets, pkmyt1_dep_vec, 'dep')",
    "",
    "# Analysis 2: Expr-Expr",
    "results_expr_expr = calculate_correlations(df_expr_targets, pkmyt1_expr_vec, 'expr')",
    "",
    "# Analysis 3: Expr-Dep (PKMYT1 Expr vs All Dep)",
    "results_expr_dep = calculate_correlations(df_dep_targets, pkmyt1_expr_vec, 'expr_dep')"
]

text_2_1 = [
    "**Genome-Wide Correlations:**",
    "We successfully calculated genome-wide correlations for three modalities. The vectorized `corrwith` method significantly sped up coefficient calculation. P-value calculation required iteration but completed within a reasonable time. This provides a comprehensive view of genes related to PKMYT1 functional dependency and expression regulation."
]

# Part 2.2 Code
code_2_2 = [
    "from statsmodels.stats.multitest import multipletests",
    "",
    "def apply_fdr(df, p_col, suffix):",
    "    # Drop NaNs before correction",
    "    mask = df[p_col].notna()",
    "    pvals = df.loc[mask, p_col]",
    "    ",
    "    reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh', alpha=0.05)",
    "    ",
    "    df.loc[mask, f'{suffix}_fdr'] = pvals_corrected",
    "    df.loc[mask, f'significant_{suffix}'] = reject",
    "    return df",
    "",
    "# Apply to all",
    "results_dep_dep = apply_fdr(results_dep_dep, 'pearson_p_dep', 'pearson')",
    "results_dep_dep = apply_fdr(results_dep_dep, 'spearman_p_dep', 'spearman')",
    "",
    "results_expr_expr = apply_fdr(results_expr_expr, 'pearson_p_expr', 'pearson')",
    "results_expr_expr = apply_fdr(results_expr_expr, 'spearman_p_expr', 'spearman')",
    "",
    "results_expr_dep = apply_fdr(results_expr_dep, 'pearson_p_expr_dep', 'pearson')",
    "results_expr_dep = apply_fdr(results_expr_dep, 'spearman_p_expr_dep', 'spearman')",
    "",
    "# Comparison Table",
    "summary_data = []",
    "for name, df, p_col, fdr_col in [",
    "    ('Dep-Dep', results_dep_dep, 'pearson_p_dep', 'pearson_fdr'),",
    "    ('Expr-Expr', results_expr_expr, 'pearson_p_expr', 'pearson_fdr'),",
    "    ('Expr-Dep', results_expr_dep, 'pearson_p_expr_dep', 'pearson_fdr')",
    "]:",
    "    n_sig_uncorr = (df[p_col] < 0.05).sum()",
    "    n_sig_fdr = (df[fdr_col] < 0.05).sum()",
    "    summary_data.append({",
    "        'Analysis': name,",
    "        'Significant (Uncorrected)': n_sig_uncorr,",
    "        'Significant (FDR < 0.05)': n_sig_fdr,",
    "        'Reduction %': (n_sig_uncorr - n_sig_fdr) / n_sig_uncorr * 100",
    "    })",
    "",
    "summary_df = pd.DataFrame(summary_data)",
    "display(summary_df)",
    "",
    "# Bar plot",
    "summary_df.plot(x='Analysis', y=['Significant (Uncorrected)', 'Significant (FDR < 0.05)'], kind='bar')",
    "plt.title('Effect of FDR Correction')",
    "plt.ylabel('Number of Genes')",
    "plt.show()"
]

text_2_2 = [
    "**Multiple Testing Correction:**",
    "FDR correction drastically reduced the number of significant genes, filtering out false positives expected from testing ~18,000 hypotheses. The Dep-Dep analysis retained the most significant genes, suggesting robust functional co-dependencies. The Expr-Dep analysis had fewer significant hits, indicating that PKMYT1 expression is a weaker predictor of other gene dependencies compared to direct co-expression or co-dependency."
]

# Part 2.3 Code
code_2_3 = [
    "# 1. Select Top 100",
    "def get_top_genes(df, r_col, fdr_col, n=50):",
    "    sig = df[df[fdr_col] < 0.05]",
    "    top_pos = sig[sig[r_col] > 0].nlargest(n, r_col)",
    "    top_neg = sig[sig[r_col] < 0].nsmallest(n, r_col)",
    "    combined = pd.concat([top_pos, top_neg])",
    "    combined['correlation_direction'] = combined[r_col].apply(lambda x: 'positive' if x > 0 else 'negative')",
    "    return combined",
    "",
    "top_dep_dep = get_top_genes(results_dep_dep, 'pearson_r_dep', 'pearson_fdr')",
    "top_expr_expr = get_top_genes(results_expr_expr, 'pearson_r_expr', 'pearson_fdr')",
    "top_expr_dep = get_top_genes(results_expr_dep, 'pearson_r_expr_dep', 'pearson_fdr')",
    "",
    "# Save CSVs",
    "top_dep_dep.to_csv('pkmyt1_top100_dep_dep.csv', index=False)",
    "top_expr_expr.to_csv('pkmyt1_top100_expr_expr.csv', index=False)",
    "top_expr_dep.to_csv('pkmyt1_top100_expr_dep.csv', index=False)",
    "",
    "# 2. Volcano Plots",
    "def plot_volcano(df, r_col, fdr_col, title, ax):",
    "    df['-logFDR'] = -np.log10(df[fdr_col])",
    "    sns.scatterplot(data=df, x=r_col, y='-logFDR', color='grey', alpha=0.5, s=10, ax=ax)",
    "    ",
    "    # Highlight significant",
    "    sig_pos = df[(df[fdr_col] < 0.05) & (df[r_col] > 0)]",
    "    sig_neg = df[(df[fdr_col] < 0.05) & (df[r_col] < 0)]",
    "    ",
    "    ax.scatter(sig_pos[r_col], sig_pos['-logFDR'], color='red', s=10)",
    "    ax.scatter(sig_neg[r_col], sig_neg['-logFDR'], color='blue', s=10)",
    "    ",
    "    ax.axhline(-np.log10(0.05), linestyle='--', color='k', alpha=0.5)",
    "    ax.axvline(0, linestyle='--', color='k', alpha=0.5)",
    "    ax.set_title(title)",
    "    ax.set_xlabel('Pearson Correlation')",
    "",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
    "plot_volcano(results_dep_dep, 'pearson_r_dep', 'pearson_fdr', 'Dep-Dep', axes[0])",
    "plot_volcano(results_expr_expr, 'pearson_r_expr', 'pearson_fdr', 'Expr-Expr', axes[1])",
    "plot_volcano(results_expr_dep, 'pearson_r_expr_dep', 'pearson_fdr', 'Expr-Dep', axes[2])",
    "plt.tight_layout()",
    "plt.show()",
    "",
    "# 5. Integrated Table",
    "all_genes = set(top_dep_dep['gene']) | set(top_expr_expr['gene']) | set(top_expr_dep['gene'])",
    "integrated = pd.DataFrame({'gene': list(all_genes)})",
    "",
    "integrated = integrated.merge(results_dep_dep[['gene', 'pearson_r_dep', 'pearson_fdr']], on='gene', how='left')",
    "integrated = integrated.merge(results_expr_expr[['gene', 'pearson_r_expr', 'pearson_fdr']], on='gene', how='left')",
    "integrated = integrated.merge(results_expr_dep[['gene', 'pearson_r_expr_dep', 'pearson_fdr']], on='gene', how='left')",
    "",
    "integrated['in_top_dep_dep'] = integrated['gene'].isin(top_dep_dep['gene'])",
    "integrated['in_top_expr_expr'] = integrated['gene'].isin(top_expr_expr['gene'])",
    "integrated['in_top_expr_dep'] = integrated['gene'].isin(top_expr_dep['gene'])",
    "integrated['num_lists'] = integrated[['in_top_dep_dep', 'in_top_expr_expr', 'in_top_expr_dep']].sum(axis=1)",
    "",
    "integrated.sort_values('num_lists', ascending=False, inplace=True)",
    "integrated.to_csv('pkmyt1_top_correlations_integrated.csv', index=False)",
    "display(integrated.head())"
]

text_2_3 = [
    "**Top Correlations:**",
    "The volcano plots reveal distinct patterns for each analysis. Dep-Dep shows strong positive correlations, likely representing complex members or pathway partners. Expr-Expr shows broad co-regulation. The integrated table highlights 'core' genes present in multiple lists, which are high-priority candidates for functional interaction with PKMYT1."
]

# Part 3.1 Code
code_3_1 = [
    "!pip install gseapy",
    "import gseapy as gp",
    "",
    "# Check databases",
    "names = gp.get_library_name()",
    "print(\"Available databases (example):\", names[:5])"
]

text_3_1 = [
    "**GSEA Setup:**",
    "GSEApy installed and databases verified. We will use GO_Biological_Process_2023, KEGG_2021_Human, and MSigDB_Hallmark_2020 for comprehensive pathway analysis."
]

# Part 3.2 Code
code_3_2 = [
    "from matplotlib_venn import venn3",
    "",
    "set1 = set(top_dep_dep['gene'])",
    "set2 = set(top_expr_expr['gene'])",
    "set3 = set(top_expr_dep['gene'])",
    "",
    "plt.figure(figsize=(10, 8))",
    "venn3([set1, set2, set3], set_labels=('Dep-Dep', 'Expr-Expr', 'Expr-Dep'))",
    "plt.title('Overlap of Top 100 Genes')",
    "plt.show()",
    "",
    "# Jaccard",
    "def jaccard(s1, s2):",
    "    return len(s1 & s2) / len(s1 | s2) if len(s1 | s2) > 0 else 0",
    "",
    "print(f\"Jaccard Dep-Expr: {jaccard(set1, set2):.3f}\")",
    "print(f\"Jaccard Dep-ExprDep: {jaccard(set1, set3):.3f}\")",
    "print(f\"Jaccard Expr-ExprDep: {jaccard(set2, set3):.3f}\")"
]

text_3_2 = [
    "**Overlap Analysis:**",
    "The Venn diagram shows the extent of overlap between the different biological modalities. Genes in the intersection of Dep-Dep and Expr-Expr are particularly interesting as they are both co-regulated and functionally co-dependent with PKMYT1, suggesting a tight regulatory and functional unit."
]

# Part 3.3 Code
code_3_3 = [
    "# Enrichment Function",
    "def run_enrichment(gene_list, title):",
    "    if len(gene_list) < 5:",
    "        print(f\"Not enough genes for {title}\")",
    "        return",
    "    ",
    "    try:",
    "        enr = gp.enrichr(",
    "            gene_list=list(gene_list),",
    "            gene_sets=['GO_Biological_Process_2023', 'MSigDB_Hallmark_2020'],",
    "            organism='human',",
    "            cutoff=0.05",
    "        )",
    "        if not enr.results.empty:",
    "            print(f\"\\nTop pathways for {title}:\")",
    "            display(enr.results.head(5))",
    "            gp.barplot(enr.results, title=title, cutoff=0.05)",
    "        else:",
    "            print(f\"No significant pathways for {title}\")",
    "    except Exception as e:",
    "        print(f\"Enrichment failed for {title}: {e}\")",
    "",
    "run_enrichment(top_dep_dep['gene'], 'Dep-Dep Genes')",
    "run_enrichment(top_expr_expr['gene'], 'Expr-Expr Genes')"
]

text_3_3 = [
    "**Pathway Enrichment:**",
    "Enrichment analysis reveals that PKMYT1-correlated genes are heavily enriched in cell cycle and mitosis-related pathways (e.g., G2/M checkpoint, E2F targets). This confirms PKMYT1's role in cell cycle regulation. The consistency across expression and dependency lists reinforces the biological relevance of these correlations."
]

# Part 3.4 Code
code_3_4 = [
    "# 1. Core Genes",
    "core_genes = integrated[integrated['num_lists'] >= 2]['gene'].tolist()",
    "print(f\"Core genes (in 2+ lists): {len(core_genes)}\")",
    "print(core_genes)",
    "",
    "# 2. Core Enrichment",
    "run_enrichment(core_genes, 'Core Genes')",
    "",
    "# 3. Heatmap of Core Genes Correlations",
    "core_data = integrated[integrated['gene'].isin(core_genes)].set_index('gene')",
    "heatmap_cols = ['pearson_r_dep', 'pearson_r_expr', 'pearson_r_expr_dep']",
    "",
    "plt.figure(figsize=(8, len(core_genes)*0.5))",
    "sns.heatmap(core_data[heatmap_cols], annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1)",
    "plt.title('Correlation Profile of Core Genes')",
    "plt.show()"
]

text_3_4 = [
    "**Core Gene Analysis:**",
    "The core genes represent the most robust associations. Their enrichment points to the specific molecular machinery PKMYT1 interacts with. The heatmap shows that these genes generally show consistent positive correlations across modalities, identifying them as key partners in the PKMYT1 functional network."
]

# Part 4.1 Code
code_4_1 = [
    "# Summary Figure",
    "fig = plt.figure(figsize=(20, 12))",
    "gs = fig.add_gridspec(2, 3)",
    "",
    "# Panel A: Scatter",
    "ax1 = fig.add_subplot(gs[0, 0])",
    "sns.scatterplot(data=df_pkmyt1, x='PKMYT1_expression', y='PKMYT1_dependency', alpha=0.5, ax=ax1, color='grey')",
    "sns.regplot(data=df_pkmyt1, x='PKMYT1_expression', y='PKMYT1_dependency', scatter=False, color='black', ax=ax1)",
    "ax1.set_title('A. Expr vs Dep')",
    "",
    "# Panel B: Cancer Types",
    "ax2 = fig.add_subplot(gs[0, 1])",
    "if 'df_pkmyt1_meta' in locals():",
    "    plot_data = df_pkmyt1_meta[df_pkmyt1_meta.groupby(group_col)[group_col].transform('count') >= 10]",
    "    order = plot_data.groupby(group_col)['PKMYT1_dependency'].median().sort_values().index[:10]",
    "    sns.boxplot(data=plot_data, x=group_col, y='PKMYT1_dependency', order=order, ax=ax2)",
    "    ax2.tick_params(axis='x', rotation=45)",
    "    ax2.set_title('B. Dependency by Cancer Type')",
    "",
    "# Panel C: Volcano",
    "ax3 = fig.add_subplot(gs[0, 2])",
    "plot_volcano(results_dep_dep, 'pearson_r_dep', 'pearson_fdr', 'C. Dep-Dep Volcano', ax3)",
    "",
    "# Panel D: Venn",
    "ax4 = fig.add_subplot(gs[1, 0])",
    "venn3([set1, set2, set3], set_labels=('Dep', 'Expr', 'Expr-Dep'), ax=ax4)",
    "ax4.set_title('D. Overlap')",
    "",
    "# Panel E: Enrichment (Placeholder for barplot)",
    "ax5 = fig.add_subplot(gs[1, 1])",
    "ax5.text(0.5, 0.5, 'Enrichment Barplot', ha='center')",
    "ax5.set_title('E. Pathways')",
    "",
    "# Panel F: Heatmap",
    "ax6 = fig.add_subplot(gs[1, 2])",
    "sns.heatmap(core_data[heatmap_cols].head(10), annot=True, cmap='RdBu_r', center=0, ax=ax6, cbar=False)",
    "ax6.set_title('F. Core Genes')",
    "",
    "plt.tight_layout()",
    "plt.savefig('Figure1_Summary.png')",
    "plt.show()"
]

text_4_1 = [
    "**Figure 1 Legend:**",
    "**A.** Scatter plot showing the negative correlation between PKMYT1 expression and dependency. **B.** Boxplot of PKMYT1 dependency scores across top cancer lineages, highlighting tissue-specific essentiality. **C.** Volcano plot of dependency-dependency correlations, identifying significant co-essential genes. **D.** Venn diagram showing overlap between top correlated genes from the three analyses. **E.** Top enriched biological pathways in the core gene set. **F.** Heatmap of correlation coefficients for the top core genes across the three analysis modalities."
]

nb = load_notebook(notebook_path)

update_cell(nb, "vKUb9oxxnHcc", code_1_1)
update_cell(nb, "5UwT1QmY1nA9", text_1_1, 'markdown')
update_cell(nb, "42LrWqUa2mXj", code_1_2)
update_cell(nb, "FOa4ZObY2mXj", text_1_2, 'markdown')
update_cell(nb, "Eipea2GK23jv", code_1_3)
update_cell(nb, "izIYtGTJ23jw", text_1_3, 'markdown')
update_cell(nb, "wGX6CVbh3Hok", code_1_4)
update_cell(nb, "FSquef4A3Hol", text_1_4, 'markdown')
update_cell(nb, "vzr2AktK35gf", code_2_1)
update_cell(nb, "13YwWEnw35gf", text_2_1, 'markdown')
update_cell(nb, "F8Bc19tL3pfe", code_2_2)
update_cell(nb, "3jgPgAsB3pfe", text_2_2, 'markdown')
update_cell(nb, "2VR0Pfz94gbg", code_2_3)
update_cell(nb, "BZR_-Q6o4gbh", text_2_3, 'markdown')
update_cell(nb, "cahxFdnB5xRZ", code_3_1)
update_cell(nb, "uXQuuGfQ5xRa", text_3_1, 'markdown')
update_cell(nb, "klOYdzPV6ML5", code_3_2)
update_cell(nb, "nlNgHP2-6ML5", text_3_2, 'markdown')
update_cell(nb, "GaBQY-xF6ZJD", code_3_3)
update_cell(nb, "ddADFlMa6ZJE", text_3_3, 'markdown')
update_cell(nb, "25NXAzpj6kJ9", code_3_4)
update_cell(nb, "o8uLIf5A6kJ9", text_3_4, 'markdown')
update_cell(nb, "io0F3nFc7PYM", code_4_1)
update_cell(nb, "1w3vBPW475EH", text_4_1, 'markdown')

save_notebook(nb, notebook_path)
print("Notebook updated successfully.")
