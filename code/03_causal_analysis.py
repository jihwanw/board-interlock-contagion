"""
Temporal Contagion: Matched Counterfactual + Firm FE Panel Regression
1. Matched counterfactual: edge가 있는 기업 쌍 vs 없는 유사 기업 쌍
2. Firm + Year FE panel regression: edge exposure → future restatement
"""
import pandas as pd, numpy as np, json, time, sys, os
from collections import defaultdict
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DATA = 'data'
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.0f}s] {msg}"); sys.stdout.flush()

# ── Load data ──
log("Loading data...")
panel = pd.read_csv(os.path.join(DATA, 'integrated_panel.csv'))
panel['cfkey'] = panel['company_fkey'].astype(str)
contagion = pd.read_csv(os.path.join(DATA, 'temporal_contagion_pairs.csv'))
edges_lc = pd.read_csv(os.path.join(DATA, 'edge_lifecycles.csv'))
log(f"Panel: {len(panel):,}, Contagion pairs: {len(contagion):,}, Edge lifecycles: {len(edges_lc):,}")

# ══════════════════════════════════════════════════════════════
# PART 1: Firm-Year Panel — Edge Exposure → Future Restatement
# ══════════════════════════════════════════════════════════════
log("\n=== PART 1: Firm-Year Panel Regression (Firm + Year FE) ===")

# Build firm-year level exposure variable from temporal contagion pairs
contagion['edge_birth_year'] = pd.to_datetime(contagion['edge_birth']).dt.year

# For each (dest, year): was this firm exposed to a restatement-origin edge?
exposure_by_firm_year = contagion.groupby(['dest', 'edge_birth_year']).agg(
    n_edges=('source_prior_restate', 'count'),
    n_exposed_edges=('source_prior_restate', 'sum'),
    any_exposed=('source_prior_restate', 'max'),
).reset_index()
exposure_by_firm_year.columns = ['cfkey', 'year', 'n_new_edges', 'n_exposed_edges', 'any_exposed']
exposure_by_firm_year['cfkey'] = exposure_by_firm_year['cfkey'].astype(str)
exposure_by_firm_year['exposure_ratio'] = (
    exposure_by_firm_year['n_exposed_edges'] / exposure_by_firm_year['n_new_edges']
)

# Merge with panel
panel_merged = panel.merge(exposure_by_firm_year, on=['cfkey', 'year'], how='left')
panel_merged['any_exposed'] = panel_merged['any_exposed'].fillna(0).astype(int)
panel_merged['n_exposed_edges'] = panel_merged['n_exposed_edges'].fillna(0)
panel_merged['exposure_ratio'] = panel_merged['exposure_ratio'].fillna(0)

log(f"Panel with exposure: {len(panel_merged):,} firm-years")
log(f"  Exposed firm-years: {(panel_merged['any_exposed']==1).sum():,} "
    f"({100*(panel_merged['any_exposed']==1).mean():.1f}%)")

# ── Firm + Year FE via demeaning (within estimator) ──
log("Running Firm + Year FE regression...")

dep_var = 'any_restatement'
indep_vars = ['any_exposed', 'n_exposed_edges', 'exposure_ratio',
              'degree', 'betweenness', 'size', 'leverage', 'roa']

reg_df = panel_merged[['gvkey', 'year', dep_var] + indep_vars].dropna()
log(f"Regression sample: {len(reg_df):,} obs, {reg_df['gvkey'].nunique():,} firms")

# Demean by firm and year
def demean_fe(df, entity_col, time_col, vars_to_demean):
    """Within transformation: subtract entity mean and time mean, add grand mean"""
    result = df.copy()
    for v in vars_to_demean:
        entity_mean = df.groupby(entity_col)[v].transform('mean')
        time_mean = df.groupby(time_col)[v].transform('mean')
        grand_mean = df[v].mean()
        result[v] = df[v] - entity_mean - time_mean + grand_mean
    return result

vars_all = [dep_var] + indep_vars
reg_demeaned = demean_fe(reg_df, 'gvkey', 'year', vars_all)

# OLS on demeaned data
from numpy.linalg import lstsq

y = reg_demeaned[dep_var].values
X = reg_demeaned[indep_vars].values
X = np.column_stack([np.ones(len(X)), X])  # intercept (should be ~0 after demeaning)

beta, residuals, rank, sv = lstsq(X, y, rcond=None)
y_hat = X @ beta
resid = y - y_hat

# Clustered standard errors (by firm)
n, k = X.shape
n_firms = reg_df['gvkey'].nunique()
n_years = reg_df['year'].nunique()
dof = n - k - n_firms - n_years  # approximate dof with FE

# Cluster-robust SE (firm-level)
firms = reg_df['gvkey'].values
unique_firms = np.unique(firms)
meat = np.zeros((k, k))
for f in unique_firms:
    mask = firms == f
    Xi = X[mask]
    ei = resid[mask]
    score = Xi.T @ np.diag(ei)
    meat += score @ score.T

bread = np.linalg.inv(X.T @ X)
V_cluster = (n_firms / (n_firms - 1)) * (n / dof) * bread @ meat @ bread
se_cluster = np.sqrt(np.diag(V_cluster))
t_stats = beta / se_cluster
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=dof))

# R-squared (within)
ss_res = np.sum(resid**2)
ss_tot = np.sum((reg_demeaned[dep_var].values - reg_demeaned[dep_var].mean())**2)
r2_within = 1 - ss_res / ss_tot

log(f"\nFirm + Year FE Regression: {dep_var}")
log(f"{'─'*65}")
log(f"{'Variable':<22} {'Coef':>10} {'SE':>10} {'t':>8} {'p':>10}")
log(f"{'─'*65}")
var_names = ['(intercept)'] + indep_vars
for i, v in enumerate(var_names):
    sig = '***' if p_values[i] < 0.01 else '**' if p_values[i] < 0.05 else '*' if p_values[i] < 0.1 else ''
    log(f"{v:<22} {beta[i]:>10.5f} {se_cluster[i]:>10.5f} {t_stats[i]:>8.3f} {p_values[i]:>9.6f} {sig}")
log(f"{'─'*65}")
log(f"R² (within): {r2_within:.4f}")
log(f"N={n:,}, Firms={n_firms:,}, Years={n_years}")

# Store regression results
reg_results = {}
for i, v in enumerate(var_names):
    reg_results[v] = {
        'coef': round(float(beta[i]), 6),
        'se': round(float(se_cluster[i]), 6),
        't': round(float(t_stats[i]), 3),
        'p': round(float(p_values[i]), 6),
    }

# ══════════════════════════════════════════════════════════════
# PART 2: Matched Counterfactual
# ══════════════════════════════════════════════════════════════
log("\n=== PART 2: Matched Counterfactual Analysis ===")

# Treatment: firms that received a new edge from a restatement-origin company
# Control: firms in the same year, same size decile, same industry (2-digit SIC proxy via gvkey)
#          that did NOT receive such an edge

# Build treatment indicator per firm-year
treatment = panel_merged[panel_merged['any_exposed'] == 1][['gvkey', 'year', 'cfkey', 'size', 'any_restatement']].copy()
control_pool = panel_merged[panel_merged['any_exposed'] == 0][['gvkey', 'year', 'cfkey', 'size', 'any_restatement']].copy()

log(f"Treatment firm-years: {len(treatment):,}")
log(f"Control pool: {len(control_pool):,}")

# Match by year + size quintile (nearest neighbor within same year)
treatment['size_q'] = treatment.groupby('year')['size'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
control_pool['size_q'] = control_pool.groupby('year')['size'].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))

matched_pairs = []
np.random.seed(42)
for (yr, sq), tg in treatment.groupby(['year', 'size_q']):
    cg = control_pool[(control_pool['year'] == yr) & (control_pool['size_q'] == sq)]
    if len(cg) == 0:
        continue
    # Random match (1:1) without replacement
    n_match = min(len(tg), len(cg))
    t_sample = tg.sample(n_match, random_state=42)
    c_sample = cg.sample(n_match, random_state=42)
    for (_, t_row), (_, c_row) in zip(t_sample.iterrows(), c_sample.iterrows()):
        matched_pairs.append({
            'year': yr,
            'treated_gvkey': t_row['gvkey'],
            'treated_restate': t_row['any_restatement'],
            'control_gvkey': c_row['gvkey'],
            'control_restate': c_row['any_restatement'],
        })

mdf = pd.DataFrame(matched_pairs)
log(f"Matched pairs: {len(mdf):,}")

if len(mdf) > 0:
    treat_rate = mdf['treated_restate'].mean()
    ctrl_rate = mdf['control_restate'].mean()
    att = treat_rate - ctrl_rate

    # McNemar's test for matched pairs
    a = ((mdf['treated_restate'] == 1) & (mdf['control_restate'] == 0)).sum()  # treat=1, ctrl=0
    b = ((mdf['treated_restate'] == 0) & (mdf['control_restate'] == 1)).sum()  # treat=0, ctrl=1
    mcnemar_chi2 = (a - b)**2 / (a + b) if (a + b) > 0 else 0
    mcnemar_p = 1 - stats.chi2.cdf(mcnemar_chi2, df=1)

    # Also simple z-test
    n_m = len(mdf)
    p_pool_m = (treat_rate + ctrl_rate) / 2
    se_m = np.sqrt(2 * p_pool_m * (1 - p_pool_m) / n_m) if p_pool_m > 0 else 1
    z_m = att / se_m
    p_z_m = 2 * (1 - stats.norm.cdf(abs(z_m)))

    log(f"\nMatched Counterfactual Results:")
    log(f"{'─'*50}")
    log(f"Treated restatement rate:  {treat_rate:.4f}")
    log(f"Control restatement rate:  {ctrl_rate:.4f}")
    log(f"ATT (difference):          {att:+.4f}")
    log(f"McNemar χ²={mcnemar_chi2:.3f}, p={mcnemar_p:.6f}")
    log(f"Z-test: z={z_m:.3f}, p={p_z_m:.6f}")

    matched_results = {
        'n_pairs': len(mdf),
        'treated_rate': round(treat_rate, 4),
        'control_rate': round(ctrl_rate, 4),
        'ATT': round(att, 4),
        'mcnemar_chi2': round(mcnemar_chi2, 3),
        'mcnemar_p': round(mcnemar_p, 6),
        'z_stat': round(z_m, 3),
        'p_z': round(p_z_m, 6),
    }
else:
    matched_results = {'error': 'No matched pairs found'}

# ── Save all results ──
all_results = {
    'panel_regression': {
        'n_obs': n, 'n_firms': n_firms, 'n_years': n_years,
        'r2_within': round(r2_within, 4),
        'coefficients': reg_results,
    },
    'matched_counterfactual': matched_results,
}

with open(os.path.join(DATA, 'causal_analysis_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

log(f"\nSaved: data/causal_analysis_results.json")
log(f"Total time: {time.time()-t0:.0f}s")
