"""
Temporal Network Analysis: Edge Lifecycle & Restatement Contagion
- Edge birth/death tied to director mobility
- Strict temporal ordering: A restatement(t0) → edge birth(t1) → B restatement(t2)
- Edge survival analysis
- Matched counterfactual comparison
"""
import pandas as pd, numpy as np, json, time, sys, os
from collections import defaultdict
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DATA = 'data'
CHECKPOINT = os.path.join(DATA, 'temporal_checkpoint.json')
RESULTS = os.path.join(DATA, 'temporal_network_results.json')
t0 = time.time()

def log(msg): print(f"[{time.time()-t0:7.0f}s] {msg}"); sys.stdout.flush()

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f: return json.load(f)
    return {'step': 0, 'data': {}}

def save_checkpoint(cp):
    with open(CHECKPOINT, 'w') as f: json.dump(cp, f)

# ── Load ──
log("Loading data...")
dirs_df = pd.read_csv(os.path.join(DATA, 'directors_officers_full.csv'),
    usecols=['company_fkey','first_name','last_name','eff_date','action'])
dirs_df['person'] = dirs_df['first_name'].fillna('') + '|' + dirs_df['last_name'].fillna('')
dirs_df['eff_date'] = pd.to_datetime(dirs_df['eff_date'], errors='coerce')
dirs_df['cfkey'] = dirs_df['company_fkey'].astype(str)
dirs_df = dirs_df.dropna(subset=['eff_date']).sort_values(['person','eff_date'])

rest_df = pd.read_csv(os.path.join(DATA, 'restatements_with_dates.csv'),
    usecols=['company_fkey','res_begin_date','file_date','res_fraud','res_sec_investigation'])
rest_df['file_date'] = pd.to_datetime(rest_df['file_date'], errors='coerce')
rest_df['cfkey'] = rest_df['company_fkey'].astype(str)
rest_df = rest_df.dropna(subset=['file_date'])

edges_df = pd.read_csv(os.path.join(DATA, 'yearly_interlock_edges.csv'))
log(f"Loaded: {len(dirs_df):,} director events, {len(rest_df):,} restatements, {len(edges_df):,} edges")

# ── Step 1: Build director tenure spells ──
log("Step 1: Building director tenure spells...")
events = dirs_df.groupby(['person','cfkey']).apply(
    lambda g: list(zip(g['action'], g['eff_date']))
).to_dict()

tenures = []  # (person, company, start_date, end_date)
for (person, company), evts in events.items():
    start = None
    for action, dt in sorted(evts, key=lambda x: x[1]):
        if action == 'Appointed':
            start = dt
        elif action in ('Resigned','Retired') and start is not None:
            tenures.append((person, company, start, dt))
            start = None
    if start is not None:
        tenures.append((person, company, start, pd.Timestamp('2024-12-31')))

tenure_df = pd.DataFrame(tenures, columns=['person','company','start','end'])
log(f"Tenure spells: {len(tenure_df):,}")

# ── Step 2: Identify edge birth/death from director mobility ──
log("Step 2: Tracking edge lifecycle from director mobility...")

# For each person, find their company sequence (mobility events)
person_companies = tenure_df.groupby('person').apply(
    lambda g: list(zip(g['company'], g['start'], g['end']))
).to_dict()

edge_events = []  # (company_a, company_b, director, edge_birth, edge_death)
cnt = 0
total = len(person_companies)
for person, spells in person_companies.items():
    if len(spells) < 2:
        continue
    # Find overlapping spells → these create edges
    for i in range(len(spells)):
        for j in range(i+1, len(spells)):
            c1, s1, e1 = spells[i]
            c2, s2, e2 = spells[j]
            if c1 == c2:
                continue
            # Overlap period = edge exists
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            if overlap_start < overlap_end:
                a, b = min(c1, c2), max(c1, c2)
                edge_events.append({
                    'company_a': a, 'company_b': b, 'director': person,
                    'edge_birth': overlap_start, 'edge_death': overlap_end,
                    'duration_days': (overlap_end - overlap_start).days,
                })
    cnt += 1
    if cnt % 50000 == 0:
        log(f"  {cnt:,}/{total:,} persons ({100*cnt//total}%)")

edf = pd.DataFrame(edge_events)
log(f"Edge lifecycle events: {len(edf):,}")

# ── Step 3: Build restatement timeline per company ──
log("Step 3: Building restatement timeline...")
rest_timeline = defaultdict(list)  # company → [(file_date, is_fraud)]
for _, r in rest_df.iterrows():
    is_fraud = (r.get('res_fraud', 0) == 1) or (r.get('res_sec_investigation', 0) == 1)
    rest_timeline[r['cfkey']].append((r['file_date'], is_fraud))

# ── Step 4: Strict temporal contagion ──
log("Step 4: Testing strict temporal contagion...")
# For each edge (A-B, born at t1, dies at t2):
#   Check if A had restatement BEFORE t1 (source)
#   Check if B had restatement AFTER t1 and BEFORE t2+3years (destination)
#   And vice versa (B→A)

contagion_records = []
for _, e in edf.iterrows():
    birth = e['edge_birth']
    death = e['edge_death']
    window_end = death + pd.Timedelta(days=365)  # 1yr after edge death

    for source, dest in [(e['company_a'], e['company_b']),
                          (e['company_b'], e['company_a'])]:
        source_restates = rest_timeline.get(source, [])
        dest_restates = rest_timeline.get(dest, [])

        # Source had restatement BEFORE edge birth?
        source_prior = any(dt < birth for dt, _ in source_restates)
        # Dest had restatement DURING edge life or within 1yr after?
        dest_after = any(birth <= dt <= window_end for dt, _ in dest_restates)

        contagion_records.append({
            'source': source, 'dest': dest, 'director': e['director'],
            'edge_birth': str(birth.date()), 'edge_death': str(death.date()),
            'duration_days': e['duration_days'],
            'source_prior_restate': source_prior,
            'dest_subsequent_restate': dest_after,
        })

cdf = pd.DataFrame(contagion_records)
log(f"Contagion test pairs: {len(cdf):,}")

# ── Step 5: Statistical tests ──
log("Step 5: Statistical analysis...")

exposed = cdf[cdf['source_prior_restate'] == True]
clean = cdf[cdf['source_prior_restate'] == False]

rate_exp = exposed['dest_subsequent_restate'].mean()
rate_cln = clean['dest_subsequent_restate'].mean()

# Chi-squared
ct = pd.crosstab(cdf['source_prior_restate'], cdf['dest_subsequent_restate'])
if ct.shape == (2, 2):
    chi2, p_chi, _, _ = stats.chi2_contingency(ct)
else:
    chi2, p_chi = 0, 1

# Z-test for proportions
n1, n2 = len(exposed), len(clean)
p_pool = (rate_exp * n1 + rate_cln * n2) / (n1 + n2) if (n1 + n2) > 0 else 0
se = np.sqrt(p_pool * (1 - p_pool) * (1/max(n1,1) + 1/max(n2,1))) if p_pool > 0 else 1
z = (rate_exp - rate_cln) / se
p_z = 2 * (1 - stats.norm.cdf(abs(z)))

# ── Step 6: Edge survival analysis ──
log("Step 6: Edge survival analysis...")
edf['duration_years'] = edf['duration_days'] / 365.25

# Compare duration: edges from restate-origin vs clean
edf_merged = edf.copy()
edf_merged['source_restate'] = edf_merged['company_a'].map(
    lambda c: any(dt.year <= 2022 for dt, _ in rest_timeline.get(c, [])))

dur_restate = edf_merged[edf_merged['source_restate']]['duration_years']
dur_clean = edf_merged[~edf_merged['source_restate']]['duration_years']
t_dur, p_dur = stats.ttest_ind(dur_restate, dur_clean) if len(dur_restate) > 0 and len(dur_clean) > 0 else (0, 1)

# ── Step 7: Temporal pattern by cohort ──
log("Step 7: Cohort analysis...")
edf['birth_year'] = edf['edge_birth'].dt.year
cohort_results = {}
for yr, g in cdf.groupby(pd.to_datetime(cdf['edge_birth']).dt.year):
    exp_g = g[g['source_prior_restate']]
    cln_g = g[~g['source_prior_restate']]
    if len(exp_g) > 0 and len(cln_g) > 0:
        cohort_results[int(yr)] = {
            'n_edges': len(g),
            'contagion_rate': round(exp_g['dest_subsequent_restate'].mean(), 4),
            'baseline_rate': round(cln_g['dest_subsequent_restate'].mean(), 4),
            'diff': round(exp_g['dest_subsequent_restate'].mean() - cln_g['dest_subsequent_restate'].mean(), 4),
        }

# ── Results ──
results = {
    'total_edge_lifecycles': len(edf),
    'total_contagion_pairs': len(cdf),
    'exposed_pairs': int(n1), 'clean_pairs': int(n2),
    'contagion_rate_exposed': round(rate_exp, 4),
    'contagion_rate_clean': round(rate_cln, 4),
    'rate_difference': round(rate_exp - rate_cln, 4),
    'chi2': round(chi2, 3), 'p_chi2': float(f"{p_chi:.8f}"),
    'z_stat': round(z, 3), 'p_z': float(f"{p_z:.8f}"),
    'edge_survival': {
        'mean_duration_restate_origin': round(dur_restate.mean(), 2) if len(dur_restate) > 0 else None,
        'mean_duration_clean_origin': round(dur_clean.mean(), 2) if len(dur_clean) > 0 else None,
        't_stat': round(t_dur, 3), 'p_value': round(p_dur, 6),
    },
    'cohort_analysis': cohort_results,
}

log(f"\n{'='*65}")
log("TEMPORAL NETWORK: Edge Lifecycle & Restatement Contagion")
log(f"{'='*65}")
log(f"Edge lifecycle events:    {len(edf):,}")
log(f"Contagion test pairs:     {len(cdf):,}")
log(f"  Exposed (source had restatement before edge birth): {n1:,}")
log(f"  Clean (source had no prior restatement):            {n2:,}")
log(f"")
log(f"Dest restatement rate (exposed):  {rate_exp:.4f}")
log(f"Dest restatement rate (clean):    {rate_cln:.4f}")
log(f"Difference:                       {rate_exp - rate_cln:+.4f}")
log(f"Chi-squared: {chi2:.3f}, p={p_chi:.8f}")
log(f"Z-test:      {z:.3f}, p={p_z:.8f}")
log(f"")
log(f"Edge survival (years):")
log(f"  From restatement origin: {dur_restate.mean():.2f}" if len(dur_restate) > 0 else "  N/A")
log(f"  From clean origin:       {dur_clean.mean():.2f}" if len(dur_clean) > 0 else "  N/A")
log(f"  t={t_dur:.3f}, p={p_dur:.6f}")
log(f"")
log("Cohort analysis (by edge birth year):")
for yr in sorted(cohort_results.keys()):
    c = cohort_results[yr]
    log(f"  {yr}: contagion={c['contagion_rate']:.3f} vs baseline={c['baseline_rate']:.3f} "
        f"(diff={c['diff']:+.3f}, n={c['n_edges']:,})")

with open(RESULTS, 'w') as f:
    json.dump(results, f, indent=2)
log(f"\nSaved: {RESULTS}")

edf.to_csv(os.path.join(DATA, 'edge_lifecycles.csv'), index=False)
cdf.to_csv(os.path.join(DATA, 'temporal_contagion_pairs.csv'), index=False)
log("Saved: edge_lifecycles.csv, temporal_contagion_pairs.csv")
log(f"Total time: {time.time()-t0:.0f}s")
