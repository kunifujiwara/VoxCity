"""Check Plotly JSON serialization format."""
import plotly.graph_objects as go
import json

fig = go.Figure()
fig.add_trace(go.Mesh3d(
    x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0],
    i=[0], j=[1], k=[2],
    color='rgb(100,100,100)'
))
j = json.loads(fig.to_json())
print(json.dumps(j['data'][0], indent=2))
