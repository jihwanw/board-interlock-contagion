# Board Interlock Networks and Accounting Risk

Replication code for:

**"The Dark Side of Connectivity: How Board Interlock Networks Are Associated with Accounting Risk Through Director Mobility"**

Jihwan Woo and Nari Kim

*Journal of Business Research* (submitted)

## Data

All data are sourced from **Wharton Research Data Services (WRDS)** and require an institutional subscription.

| Dataset | WRDS Table | Records |
|---------|-----------|---------|
| Directors & Officers | `audit.feed17_director_and_officer_changes` | 561,306 |
| Restatements | `audit.feed39_financial_restatements` | 28,662 events |
| Auditor-Company | `audit.feed01_audit_opinions` | 71,480 |
| Compustat Financials | `comp.funda` | 48,110 firm-years |

**Data are not included in this repository due to WRDS licensing restrictions.** To obtain the data, run `code/00_download_wrds_data.py` with valid WRDS credentials.

## Replication

### Prerequisites

```bash
pip install wrds pandas numpy networkx scipy matplotlib
```

### Steps

```bash
# 1. Download data from WRDS (requires credentials)
python code/00_download_wrds_data.py

# 2. Build yearly interlock networks and edge lifecycles
python code/01_build_edge_lifecycles.py

# 3. Temporal contagion analysis (H1, H3, H4)
python code/02_temporal_contagion.py

# 4. Firm FE regression and matched counterfactual (H2)
python code/03_causal_analysis.py

# 5. Robustness tests (strict fraud, multiple windows, director roles, same-auditor, restatement types)
python code/04_robustness.py
```

All scripts include checkpoint functionality. If interrupted, re-running will resume from the last completed step.

### Output

Results are saved as JSON files in `data/`:
- `dynamic_network_results.json` — Edge birth analysis
- `temporal_network_results.json` — Temporal contagion (H1, H3, H4)
- `causal_analysis_results.json` — Firm FE and matching (H2)
- `robustness_results.json` — All robustness tests
- `community_detection_results.json` — Community structure (H5)
- `reviewer_response_results.json` — Director roles, same-auditor, restatement types

## Key Findings

| Hypothesis | Finding | p-value |
|-----------|---------|---------|
| H1: Temporal contagion | +7.0 pp (exposed vs clean) | < 0.001 |
| H2: Dose-response | +0.25 pp per exposed edge (Firm + Ind×Year FE) | < 0.001 |
| H3: Temporal decline | +12.9 pp (2010-14) → +2.1 pp (2020-23) | < 0.001 |
| H4: Edge survival | -0.95 years (contaminated vs clean) | < 0.001 |
| H5: Bridge contagion | +3.7 pp (cross- vs within-community) | < 0.001 |

## License

Code: MIT License

Data: Not included. Available from [WRDS](https://wrds-www.wharton.upenn.edu/) with institutional subscription.

## Citation

```bibtex
@article{woo2026darkside,
  title={The Dark Side of Connectivity: How Board Interlock Networks Are Associated with Accounting Risk Through Director Mobility},
  author={Woo, Jihwan and Kim, Nari},
  journal={Journal of Business Research},
  year={2026},
  note={Submitted}
}
```
