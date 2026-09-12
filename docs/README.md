# fastsae Documentation Site

Documentation + benchmark results for the `fastsae` R package.

## Preview locally

```bash
cd site
python3 -m http.server 8000
# Open: http://localhost:8000
```

## Deploy to GitHub Pages

```bash
# 1. Clone your GitHub Pages repo (if not already)
git clone https://github.com/ridsonap/ridsonap.github.io.git
cd ridsonap.github.io

# 2. Copy site files
cp /path/to/fastsaebench/site/index.html         .
cp /path/to/fastsaebench/site/benchmark-data.json .

# 3. Commit and push
git add index.html benchmark-data.json
git commit -m "Update fastsae documentation"
git push origin main

# 4. Site live at: https://ridsonap.github.io/
```

Or use the automated script:

```bash
cd ridsonap.github.io
bash /path/to/fastsaebench/deploy_fastsae.sh
```

## Site contents

| File | Description |
|------|-------------|
| `index.html` | Full documentation site with sidebar nav |
| `benchmark-data.json` | Raw benchmark data for charts |

## Building benchmark data

```bash
python3 /path/to/fastsaebench/generate_data.R
# Then rebuild index.html:
python3 /tmp/build_html.py
```

## Benchmark data

The benchmark compares `fastsae` vs `sae` vs `emdi` across:
- **n = 30, 50, 100, 250, 500, 1000** areas
- **EBLUP** and **SEBLUP** algorithms
- Time (median), memory, and throughput (iterations/sec)

Key results:
| n | vs sae (EBLUP) | vs emdi (EBLUP) |
|---|----------------|-----------------|
| 30 | 3× | 19× |
| 100 | 6× | 92× |
| 500 | 140× | 4,178× |
| 1000 | 364× | 12,343× |
