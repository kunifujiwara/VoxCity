"""Save a sample of figure JSON to a file for inspection."""
import numpy as np
import sys, os, json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from voxcity.visualizer import visualize_voxcity_plotly

# Create a tiny test voxel array
vox = np.zeros((5, 5, 3), dtype=np.int8)
vox[1:4, 1:4, 0] = 11   # ground: developed space
vox[2, 2, 1] = -3        # building
vox[2, 2, 2] = -3        # building
vox[3, 3, 1] = -2        # tree

fig = visualize_voxcity_plotly(vox, meshsize=5.0, show=False, return_fig=True)
if fig is None:
    print("ERROR: fig is None")
    sys.exit(1)

fig_json = fig.to_json()
parsed = json.loads(fig_json)
traces = parsed.get('data', [])
print(f"Total traces: {len(traces)}")
for i, t in enumerate(traces):
    tp = t.get('type', '?')
    xlen = len(t.get('x', [])) if isinstance(t.get('x'), list) else 'N/A'
    ilen = len(t.get('i', [])) if isinstance(t.get('i'), list) else 'N/A'
    print(f"  [{i}] type={tp}, x_len={xlen}, i_len={ilen}, color={t.get('color','')}, name={t.get('name','')}")
    # Print first few values of x, y, z
    for key in ['x', 'y', 'z']:
        vals = t.get(key, [])
        if isinstance(vals, list) and len(vals) > 0:
            print(f"       {key}[0:3] = {vals[:3]}  (type of first: {type(vals[0]).__name__})")

# Save first trace for full inspection
if traces:
    with open(os.path.join(os.path.dirname(__file__), 'debug_first_trace.json'), 'w') as f:
        json.dump(traces[0], f, indent=2)
    print(f"\nSaved first trace to debug_first_trace.json")
