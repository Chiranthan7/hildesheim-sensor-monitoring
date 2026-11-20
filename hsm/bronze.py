"""Bronze layer: every raw row, verbatim (all columns as text), plus a stable raw_row_id and a parsed UTC timestamp.
Nothing is dropped or altered. Optional cutoff simulates the information available at a given moment."""
import glob, hashlib, os, duckdb, pandas as pd

SOURCES = {'devices': 'devices_raw.csv', 'soil': 'bodenfeuchte_raw.csv', 'weather': 'weather_raw.csv'}

def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''): h.update(b)
    return h.hexdigest()

def find_gateway_file(raw_dir):
    hits = sorted(glob.glob(os.path.join(raw_dir, '*Gateway_Standorte*.csv')))
    if not hits: raise FileNotFoundError('gateway list not found')
    return hits[0]

def build(raw_dir, out_dir, cutoff=None):
    os.makedirs(out_dir, exist_ok=True)
    con = duckdb.connect(); manifest = {}
    for src, fn in SOURCES.items():
        path = os.path.join(raw_dir, fn)
        manifest[src] = dict(file=fn, sha256=sha256(path), bytes=os.path.getsize(path))
        q = f"""SELECT '{src}:' || CAST(row_number() OVER () + 1 AS VARCHAR) AS raw_row_id, *,
                 CAST(strptime(substr(time_index, 1, 26), '%Y-%m-%d %H:%M:%S.%f') AS TIMESTAMP) AS t
                FROM read_csv('{path}', all_varchar=true, header=true, quote='"', escape='"', sample_size=-1, strict_mode=true)"""
        if cutoff is not None:
            q = f"SELECT * FROM ({q}) WHERE t < TIMESTAMP '{pd.Timestamp(cutoff)}'"
        con.execute(f"COPY ({q}) TO '{out_dir}/bronze_{src}.parquet' (FORMAT parquet)")
        n_raw = sum(1 for _ in open(path, 'rb')) - 1   # physical lines minus header (no embedded newlines in these exports; checked below)
        n_out = con.execute(f"SELECT count(*), max(t) FROM '{out_dir}/bronze_{src}.parquet'").fetchone()
        manifest[src].update(raw_rows=int(n_raw), bronze_rows=int(n_out[0]), export_end=str(n_out[1]))
    gw = find_gateway_file(raw_dir)
    manifest['gateways'] = dict(file=os.path.basename(gw), sha256=sha256(gw))
    g = pd.read_csv(gw, encoding='utf-8-sig', dtype=str); g.insert(0, 'raw_row_id', [f'gateways:{i + 2}' for i in range(len(g))])
    g.to_parquet(f'{out_dir}/bronze_gateways.parquet', index=False)
    return manifest
