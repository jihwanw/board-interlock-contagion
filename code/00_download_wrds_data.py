"""
00_download_wrds_data.py
Download all required data from WRDS for replication.
Requires: pip install wrds pandas
Requires: Valid WRDS institutional subscription
"""
import wrds
import pandas as pd
import os

OUTPUT = 'data'
os.makedirs(OUTPUT, exist_ok=True)

print("Connecting to WRDS...")
db = wrds.Connection()  # Will prompt for credentials

# ── 1. Directors & Officers ──
print("1/4 Downloading directors & officers...")
directors = db.raw_sql("""
    SELECT company_fkey, first_name, last_name, eff_date, action,
           is_ceo, is_cfo, is_bdmem_pers, is_chair, is_c_level,
           is_fin_pers, is_op_pers, is_legal, is_exec_vp,
           is_president, is_coo, is_secretary, is_cont
    FROM audit.feed17_director_and_officer_changes
    ORDER BY company_fkey, eff_date
""")
directors.to_csv(f'{OUTPUT}/directors_officers_full.csv', index=False)
print(f"  Saved: {len(directors):,} records")

# ── 2. Restatements ──
print("2/4 Downloading restatements...")
restatements = db.raw_sql("""
    SELECT company_fkey, file_date, res_begin_date, res_end_date,
           res_accounting, res_adverse, res_fraud, res_sec_investigation,
           matchqu_balsh_assets, matchqu_incmst_netinc_ttm,
           matchqu_incmst_rev_ttm
    FROM audit.feed39_financial_restatements
    WHERE file_date IS NOT NULL
    ORDER BY company_fkey, file_date
""")
restatements.to_csv(f'{OUTPUT}/restatements_with_dates.csv', index=False)
print(f"  Saved: {len(restatements):,} records")

# ── 3. Auditor-Company relationships ──
print("3/4 Downloading auditor data...")
auditors = db.raw_sql("""
    SELECT company_fkey, auditor_key, auditor_name, event_type
    FROM audit.feed01_audit_opinions
    WHERE auditor_key IS NOT NULL
    ORDER BY company_fkey
""")
auditors.to_csv(f'{OUTPUT}/auditor_company.csv', index=False)
print(f"  Saved: {len(auditors):,} records")

# ── 4. Compustat financials ──
print("4/4 Downloading Compustat financials...")
compustat = db.raw_sql("""
    SELECT gvkey, fyear AS year, at AS assets, lt AS liabilities,
           ceq AS equity, revt AS revenue, ni AS net_income,
           oancf, csho * prcc_f AS mktcap, csho AS shares,
           prcc_f AS price
    FROM comp.funda
    WHERE fyear BETWEEN 2004 AND 2023
      AND indfmt = 'INDL' AND datafmt = 'STD'
      AND popsrc = 'D' AND consol = 'C'
      AND at > 0
    ORDER BY gvkey, fyear
""")
compustat.to_csv(f'{OUTPUT}/compustat_financials.csv', index=False)
print(f"  Saved: {len(compustat):,} records")

db.close()
print("\nAll data downloaded. Run 01_build_edge_lifecycles.py next.")
