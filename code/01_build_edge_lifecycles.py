"""
Dynamic Network Analysis: Director-Induced Edge Births & Fraud Contagion
- Checkpoint: saves after each year → resumes from last completed year on restart
- Optimized: set-based lookups instead of nested loops
"""
import pandas as pd, numpy as np, json, time, sys, os
from collections import defaultdict
from scipy import stats
import warnings; warnings.filterwarnings('ignore')

DATA = 'data'
CHECKPOINT = os.path.join(DATA, 'dynamic_checkpoint.json')
RESULTS = os.path.join(DATA, 'dynamic_network_results.json')
YEARS = range(2005, 2023)  # 2005-2022
t0 = time.time()

def log(msg): print(f"[{time.time()-t0:7.0f}s] {msg}"); sys.stdout.flush()

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            cp = json.load(f)
        log(f"Checkpoint loaded: {len(cp['completed_years'])} years done ({cp['completed_years']})")
        return cp
    return {'completed_years': [], 'year_results': {}, 'edge_births_all': []}

def save_checkpoint(cp):
    with open(CHECKPOINT, 'w') as f:
        json.dump(cp, f, indent=2)

# ── Load data ──
log("Loading data...")
dirs_df = pd.read_csv(os.path.join(DATA, 'directors_officers_full.csv'),
                       usecols=['company_fkey','first_name','last_name','eff_date','action','is_bdmem_pers'])
edges_df = pd.read_csv(os.path.join(DATA, 'yearly_interlock_edges.csv'))
rest_df = pd.read_csv(os.path.join(DATA, 'restatements_with_dates.csv'),
                       usecols=['company_fkey','res_begin_date','res_end_date','file_date',
                                'res_fraud','res_sec_investigation','res_adverse','res_accounting'])
log(f"Data loaded. Directors: {len(dirs_df):,}, Edges: {len(edges_df):,}, Restatements: {len(rest_df):,}")

# ── Build yearly edge sets ──
log("Building yearly edge sets...")
yearly_edges = {}
for y, g in edges_df.groupby('year'):
    yearly_edges[y] = set(zip(g['company_a'].astype(str), g['company_b'].astype(str)))
log("Done.")

# ── Build director tenures ──
log("Building director tenures...")
dirs_df['person'] = dirs_df['first_name'].fillna('') + '|' + dirs_df['last_name'].fillna('')
dirs_df['eff_date'] = pd.to_datetime(dirs_df['eff_date'], errors='coerce')
dirs_df['year'] = dirs_df['eff_date'].dt.year
dirs_df['cfkey'] = dirs_df['company_fkey'].astype(str)
dirs_df = dirs_df.dropna(subset=['eff_date']).sort_values(['person','cfkey','eff_date'])

# Build tenures: each (person, company) → list of (start_year, end_year)
tenures = defaultdict(list)  # (person, company) → [(start, end), ...]
active = {}  # (person, company) → start_year
cnt = 0
for _, r in dirs_df.iterrows():
    key = (r['person'], r['cfkey'])
    if r['action'] == 'Appointed':
        if key not in active:
            active[key] = r['year']
    elif r['action'] == 'Resigned' or r['action'] == 'Retired':
        if key in active:
            tenures[key].append((active.pop(key), r['year']))
        else:
            tenures[key].append((r['year'] - 3, r['year']))  # assume 3yr if no start
    cnt += 1
    if cnt % 100000 == 0:
        log(f"  Tenures: {cnt:,}/{len(dirs_df):,} ({100*cnt//len(dirs_df)}%)")

# Close open tenures (still active → assume through 2024)
for key, start in active.items():
    tenures[key].append((start, 2024))
log(f"Tenures built: {len(tenures):,}")

# ── Build per-year indices ──
log("Building per-year company→directors and director→companies indices...")
year_comp_dirs = defaultdict(lambda: defaultdict(set))  # year → company → {directors}
year_dir_comps = defaultdict(lambda: defaultdict(set))  # year → director → {companies}

for (person, company), spans in tenures.items():
    for start, end in spans:
        for y in range(max(start, 2004), min(end, 2024) + 1):
            year_comp_dirs[y][company].add(person)
            year_dir_comps[y][person].add(company)
log("Indices built.")

# ── Build restatement lookup ──
log("Building restatement lookup...")
rest_df['file_date'] = pd.to_datetime(rest_df['file_date'], errors='coerce')
rest_df['file_year'] = rest_df['file_date'].dt.year
rest_df['cfkey'] = rest_df['company_fkey'].astype(str)
# company → set of years with any restatement filed
restate_years = defaultdict(set)
fraud_years = defaultdict(set)  # strict fraud only
for _, r in rest_df.iterrows():
    if pd.notna(r['file_year']):
        restate_years[r['cfkey']].add(int(r['file_year']))
        if r.get('res_fraud', 0) == 1 or r.get('res_sec_investigation', 0) == 1:
            fraud_years[r['cfkey']].add(int(r['file_year']))
log(f"Restatement companies: {len(restate_years):,}, Fraud companies: {len(fraud_years):,}")

# ── STEP 1: Director-induced edge births with checkpoint ──
cp = load_checkpoint()
completed = set(cp['completed_years'])

log(f"STEP 1: Finding director-induced edge births ({len(YEARS)-len(completed)} years remaining)...")

for y in YEARS:
    if y in completed:
        continue
    yt = time.time()
    prev_edges = yearly_edges.get(y - 1, set())
    curr_edges = yearly_edges.get(y, set())
    new_edges = curr_edges - prev_edges

    births = []
    comp_dirs_y = year_comp_dirs[y]
    comp_dirs_prev = year_comp_dirs[y - 1]

    for a, b in new_edges:
        # Directors shared between a and b in year y
        dirs_a = comp_dirs_y.get(a, set())
        dirs_b = comp_dirs_y.get(b, set())
        shared = dirs_a & dirs_b
        if not shared:
            continue

        # Which of these are NEW shared directors (not shared in y-1)?
        dirs_a_prev = comp_dirs_prev.get(a, set())
        dirs_b_prev = comp_dirs_prev.get(b, set())
        prev_shared = dirs_a_prev & dirs_b_prev
        new_shared = shared - prev_shared

        for d in new_shared:
            # Determine which company the director moved TO
            was_at_a = a in year_dir_comps[y - 1].get(d, set())
            was_at_b = b in year_dir_comps[y - 1].get(d, set())
            if was_at_a and not was_at_b:
                origin, dest = a, b
            elif was_at_b and not was_at_a:
                origin, dest = b, a
            else:
                origin, dest = a, b  # both new or both old

            # Check fraud history at origin
            origin_had_restate = any(ry < y for ry in restate_years.get(origin, set()))
            origin_had_fraud = any(ry < y for ry in fraud_years.get(origin, set()))

            births.append({
                'year': y, 'company_a': a, 'company_b': b,
                'director': d, 'origin': origin, 'destination': dest,
                'origin_prior_restate': origin_had_restate,
                'origin_prior_fraud': origin_had_fraud,
            })

    # Destination future restatement (within 3 years)
    for b in births:
        dest = b['destination']
        b['dest_future_restate'] = any(y < ry <= y + 3 for ry in restate_years.get(dest, set()))
        b['dest_future_fraud'] = any(y < ry <= y + 3 for ry in fraud_years.get(dest, set()))

    # Save year summary
    n_births = len(births)
    n_from_restate = sum(1 for b in births if b['origin_prior_restate'])
    n_dest_restate = sum(1 for b in births if b['dest_future_restate'])
    n_from_restate_dest_restate = sum(1 for b in births if b['origin_prior_restate'] and b['dest_future_restate'])
    n_clean_dest_restate = sum(1 for b in births if not b['origin_prior_restate'] and b['dest_future_restate'])

    cp['year_results'][str(y)] = {
        'new_edges': len(new_edges),
        'director_induced_births': n_births,
        'from_restate_origin': n_from_restate,
        'dest_future_restate': n_dest_restate,
        'from_restate_AND_dest_restate': n_from_restate_dest_restate,
        'from_clean_AND_dest_restate': n_clean_dest_restate,
        'contagion_rate': n_from_restate_dest_restate / max(n_from_restate, 1),
        'baseline_rate': n_clean_dest_restate / max(n_births - n_from_restate, 1),
    }
    cp['edge_births_all'].extend(births)
    cp['completed_years'].append(y)
    completed.add(y)
    save_checkpoint(cp)

    elapsed = time.time() - yt
    log(f"  Year {y} ({len(completed)}/{len(YEARS)}): "
        f"{len(new_edges):,} new edges, {n_births:,} director-induced, "
        f"contagion={cp['year_results'][str(y)]['contagion_rate']:.3f} vs "
        f"baseline={cp['year_results'][str(y)]['baseline_rate']:.3f} "
        f"[{elapsed:.0f}s]")

# ── STEP 2: Aggregate analysis ──
log("STEP 2: Aggregate contagion analysis...")
all_births = cp['edge_births_all']
df = pd.DataFrame(all_births)

if len(df) > 0:
    # Overall contagion test
    exposed = df[df['origin_prior_restate'] == True]
    clean = df[df['origin_prior_restate'] == False]
    rate_exposed = exposed['dest_future_restate'].mean() if len(exposed) > 0 else 0
    rate_clean = clean['dest_future_restate'].mean() if len(clean) > 0 else 0

    # Chi-squared test
    ct = pd.crosstab(df['origin_prior_restate'], df['dest_future_restate'])
    chi2, p_chi, _, _ = stats.chi2_contingency(ct) if ct.shape == (2, 2) else (0, 1, 0, None)

    # Proportions z-test
    n1, n2 = len(exposed), len(clean)
    p1, p2 = rate_exposed, rate_clean
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2) if (n1 + n2) > 0 else 0
    se = np.sqrt(p_pool * (1 - p_pool) * (1/max(n1,1) + 1/max(n2,1))) if p_pool > 0 else 1
    z = (p1 - p2) / se if se > 0 else 0
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))

    results = {
        'total_director_induced_births': len(df),
        'from_restate_origin': len(exposed),
        'from_clean_origin': len(clean),
        'contagion_rate_exposed': round(rate_exposed, 4),
        'contagion_rate_clean': round(rate_clean, 4),
        'rate_difference': round(rate_exposed - rate_clean, 4),
        'chi2': round(chi2, 3), 'p_chi2': round(p_chi, 6),
        'z_stat': round(z, 3), 'p_z': round(p_z, 6),
        'yearly': cp['year_results'],
    }

    log(f"\n{'='*60}")
    log(f"RESULTS: Director-Induced Edge Births & Fraud Contagion")
    log(f"{'='*60}")
    log(f"Total director-induced births: {len(df):,}")
    log(f"From restatement origin:       {len(exposed):,} ({100*len(exposed)/len(df):.1f}%)")
    log(f"From clean origin:             {len(clean):,}")
    log(f"")
    log(f"Dest restatement rate (exposed):  {rate_exposed:.4f}")
    log(f"Dest restatement rate (clean):    {rate_clean:.4f}")
    log(f"Difference:                       {rate_exposed - rate_clean:+.4f}")
    log(f"Chi-squared: {chi2:.3f}, p={p_chi:.6f}")
    log(f"Z-test:      {z:.3f}, p={p_z:.6f}")

    with open(RESULTS, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS}")

    # Save edge births as CSV
    df.to_csv(os.path.join(DATA, 'director_edge_births.csv'), index=False)
    log(f"Edge births saved to data/director_edge_births.csv")
else:
    log("No director-induced edge births found.")

log(f"\nTotal time: {time.time()-t0:.0f}s")
