# MEDUSA 2026 Web Service

Dockerized web UI for the MEDUSA 2026 pipeline: load a **real FT-ICR mass
spectrum** (mzXML, up to ~8 M points / 107 MB) and run all 7 notebook steps
with interactive Plotly plots:

1. **Load** — pick a file from a server/NFS folder or upload from the browser
2. **Deisotoping** — group peaks into isotopic distributions, predict charges
3. **Element classification** — Transformer probabilities per ion
4. **Knee detection** — automatic element-probability threshold (any element)
5. **Highlight** — ions above the threshold on the spectrum
6. **Molecular formula determination** — ranked candidates, editable element limits
7. **Compare** — chosen formula vs the experimental spectrum

Behavior mirrors `MEDUSA2026/research/formula_analysis_examples/entry.ipynb`
(defaults are the notebook values; every parameter is adjustable in the UI).

---

## Prerequisites

- Docker (engine + Compose v2). Nothing else — the image is self-contained
  (Python 3.9, CPU torch 1.12, pyopenms 3.2, catboost 1.2.8, the pre-trained
  models baked in).

## Build & run

From this directory (`medusa_web/`, which sits next to the `MEDUSA2026/` repo):

```bash
docker compose up --build
# open http://localhost:8000
```

The compose file mounts a spectra folder read-only at `/data/spectra`:

```yaml
volumes:
  - ${SPECTRA_DIR:-./spectra}:/data/spectra:ro
```

So point `SPECTRA_DIR` at your NFS/local folder (path or a `.env` file next to
the compose file):

```bash
SPECTRA_DIR=/mnt/nfs/medusa_spectra docker compose up --build
# or
echo "SPECTRA_DIR=/mnt/nfs/medusa_spectra" > .env
docker compose up --build
```

`SPECTRA_DIR` may be nested — up to 3 directory levels are listed in the UI.
Uploaded files are kept in a named volume (`medusa_uploads`, mounted at
`/data/uploads`).

### Laptop usage

No server needed: copy (or git clone) the parent directory of `medusa_web/`
(the one that also contains `MEDUSA2026/`), put your spectra in a folder and:

```bash
cd medusa_web
SPECTRA_DIR=../my_spectra docker compose up --build
# open http://localhost:8000
```

First build pulls ~2.5 GB of wheels (torch CPU 776 MB, pyopenms, catboost) —
plan for a few minutes on a fast line. After that the image is cached.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | host port published by compose |
| `SPECTRA_DIR` | `/data/spectra` | folder listed in the UI (mounted read-only) |
| `UPLOAD_DIR` | `/data/uploads` | where uploads are stored |
| `MAX_UPLOAD_MB` | `1024` | upload size limit (413 above it) |
| `MAX_ACTIVE_SESSIONS` | `8` | 409 when reached — delete a session first |
| `MAX_CONCURRENT_CPU_JOBS` | `2` | heavy steps in parallel; others get `409 job slot busy, retry` |
| `CGB_MODEL` | baked `/app/models/charge1_big_optuna150.pkl` | deisotoper (CatBoost) path |
| `TRANSFORMER_CKPT` | baked `/app/models/transfomer_classifier.ckpt` | Transformer ckpt path |

## Usage walkthrough

1. **Load**: *Server files* tab → click a file (or drop an mzXML onto *Upload*).
   The header shows session id + file name; the viewer shows a decimated
   overview (≤ 2500 points per response — the 8 M-point scan stays smooth).
2. **Deisotoping**: defaults are the notebook values (`adaptive`, z_max 3,
   threshold 0.15, delta 0.007, min_distance 0.01, n1 2, n2 6) → *Run*.
   Toggle *ions on spectrum* for per-ion coloring; the legend lists
   `ion → m/z / charge`.
3. **Elements**: symbol (default `Ir`, validated) → *Run* — the right-hand
   ions table shows P(Ir) per ion (sortable; row click selects the ion).
4. **Threshold**: knee chart (sorted probabilities, knee marker) + auto/manual
   slider → *Apply threshold*.
5. **Highlight**: red layer for ions above the threshold; *Zoom to highlighted
   ions*.
6. **Formulas**: pick the ion (table click or number), edit the element-limit
   table (presets: `ir_system`, `pubchem10`, `empty`), set ±ppm and workers →
   *Run*. Ranked table (rank, formula, mass, Δppm, cosine); **CSV** exports it;
   row click prefills step 7.
7. **Compare**: *Run* → experimental window vs theoretical isotope pattern with
   Δppm, cosine distance and matched-peak %.

The viewer zooms/pans live (150 ms debounce, refetches the window). Hover
always shows exact point data. A small log area shows engine messages,
including per-ion skip warnings (the known `empty peak_indices` bug never
crashes a session — the ion is skipped with a warning).

## Tests

Test data: `tests/data/` is a slice (8 samples) of
`MEDUSA2026/data/formula_determination_test/test.h5` + `test.pkl`
(synthetic single-molecule FT-ICR-like spectra, 100k points, GT formulas).

```bash
# inside the image (models baked in):
docker compose run --rm web pytest medusa_web/tests -q

# in the dev venv (Python 3.9, from the PARENT of medusa_web/):
pytest medusa_web/tests -q
```

Covers: downsample properties (synthetic + real window), knee on
known-shape vectors, full pipeline smoke (deisotope → elements → knee →
formulas, GT formula present), and a TestClient round-trip through every
endpoint (upload + folder sources, errors, session limit).

## Troubleshooting

- **Port in use** — run on another port: `PORT=8010 docker compose up --build`
  (the in-container port stays 8000).
- **"upload too large: limit is 1024 MB"** — raise `MAX_UPLOAD_MB`.
- **`job slot busy, retry` (409)** — another heavy step is running
  (`MAX_CONCURRENT_CPU_JOBS`); wait and press *Run* again, or raise the limit.
- **"max active sessions reached" (409)** — press *New session* / delete an
  old session.
- **Models missing** (503 on deisotope/elements) — check the container logs
  and the `CGB_MODEL` / `TRANSFORMER_CKPT` paths; `GET /healthz` reports
  which models are loaded.
- **`spectra folder empty`** — verify the host path in `${SPECTRA_DIR}` and
  that the mount is readable inside the container
  (`docker compose exec web ls /data/spectra`).

## Notes & limitations (v1)

- No authentication (university LAN), one uvicorn worker (in-memory state).
- One scan per session (multi-scan files: scan 1 is used, `n_scans` shown).
- No job queue: long steps respond synchronously with a spinner; saturated
  callers get 409.
- CPU only.
