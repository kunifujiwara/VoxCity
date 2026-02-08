"""Quick debug script to inspect the figure JSON structure from the backend."""
import requests, json

r = requests.get('http://localhost:8000/api/model/preview', timeout=120)
d = r.json()
fj = d.get('figure_json', '{}')
fig = json.loads(fj)
traces = fig.get('data', [])
print(f'Number of traces: {len(traces)}')
for i, t in enumerate(traces[:10]):
    tp = t.get('type', '?')
    keys = sorted(t.keys())
    xlen = len(t.get('x', [])) if isinstance(t.get('x'), list) else 'N/A'
    ilen = len(t.get('i', [])) if isinstance(t.get('i'), list) else 'N/A'
    col = str(t.get('color', ''))[:60]
    name = t.get('name', '')
    print(f'  [{i}] type={tp}, name={name}, x_len={xlen}, i_len={ilen}, color={col}')
    print(f'       keys={keys}')
