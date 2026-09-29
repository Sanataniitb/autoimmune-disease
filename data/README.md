# Input data

Expression files are **not committed** to this repository (`data/*` is git-ignored).
Put them here under the file names used in `configs/`, or pass another path with
`autoimmune-ml run <config> --data <file>`.

| Config | Expected file | Source |
|---|---|---|
| `configs/rheumatoid_arthritis.yaml` | `Rheumatoid Arthritis ML Input 56 features 106 samples.xlsx` | RA RNA-seq: 56 genes × 106 samples |
| `configs/sle.yaml` | `GSE116006_SLE_N ml input.csv` | [GEO GSE116006](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116006) + healthy controls: 100 genes × 216 samples |
| `configs/t1d.yaml` | `T1D ML Input manual feature 167.xlsx` | T1D RNA-seq: 166 genes × 180 samples |
| `configs/ankylosing_spondylitis.yaml` | `Spondylitis_ml input_80 features.csv` | AS RNA-seq: 83 genes × 66 samples |
| `configs/synthetic_demo.yaml` | `synthetic_demo.csv` | made by `autoimmune-ml synthetic` |

## Format

A CSV, TSV or Excel table with **genes as rows and samples as columns**:

| Gene | SLE1_W0 | SLE2_W0 | … | healthy_34 | healthy_35 |
|---|---|---|---|---|---|
| ABCB1 | 3.84 | 1.73 | … | 2.91 | 2.87 |
| BANK1 | 5.85 | 4.30 | … | 3.84 | 3.98 |

- The gene symbol column is named `Gene` (change with `gene_column`).
- Sample labels come from the column name: names starting with
  `positive_prefix` are disease (1), every other sample is a control (0).
- Values may be raw counts (`log_transform: true`) or already log-scaled
  (`log_transform: false`).
- Duplicate gene rows are averaged; non-numeric cells are imputed with the gene median.
  Both are logged as warnings.

If several samples come from the same patient (e.g. `SLE1_W0`, `SLE1_W12`), set
`group_regex` in the config so all of a patient's samples are kept in the same CV fold.
