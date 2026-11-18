"""Folium map builder (also used by the API/UI). Coordinates: Folium expects [lat, lon]; raw files are lon,lat."""
import numpy as np, folium
def arcs_latlon(geo, arc_ids):
    A = geo.inv.transform(geo.A[arc_ids, 0], geo.A[arc_ids, 1]); B = geo.inv.transform(geo.B[arc_ids, 0], geo.B[arc_ids, 1])
    return [[(A[1][i], A[0][i]), (B[1][i], B[0][i])] for i in range(len(arc_ids))]
def route_polyline(geo, arc_seq):
    """Connected polyline of an arc sequence (start of each arc, plus end of last)."""
    arc_seq = np.asarray(arc_seq); arc_seq = arc_seq[arc_seq >= 0]
    if len(arc_seq) == 0: return []
    pts = np.vstack([geo.A[arc_seq], geo.B[arc_seq[-1]][None]]); lon, lat = geo.inv.transform(pts[:, 0], pts[:, 1])
    return list(zip(lat, lon))
def make_map(geo, track, ref_route=None, preds=None, net_radius=150.0, max_pts=1500, title="PathSnap"):
    """preds: dict name -> (arc array per obs, colour). Layers: GPS, network, reference, each prediction."""
    clat, clon = track.lat.mean(), track.lon.mean()
    m = folium.Map(location=[clat, clon], zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    # network near the track only (keeps HTML small)
    import shapely
    pts = shapely.points(geo.P[:: max(1, len(geo.P) // 400)])
    near = np.unique(geo.tree().query(pts, predicate="dwithin", distance=net_radius)[1])
    fg_net = folium.FeatureGroup("Road network (arcs near track)", show=True)
    for a, ll in zip(near, arcs_latlon(geo, near)): folium.PolyLine(ll, color="#8a8a8a", weight=2, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)
    fg_gps = folium.FeatureGroup("GPS observations (blue)", show=True)
    step = max(1, len(track) // max_pts)
    for la, lo, t in zip(track.lat.values[::step], track.lon.values[::step], track.t.values[::step]):
        folium.CircleMarker([la, lo], radius=2, color="#1f5fff", fill=True, fill_opacity=0.8, weight=0, tooltip=f"t={t:.0f}s").add_to(fg_gps)
    fg_gps.add_to(m)
    if ref_route is not None:
        fg = folium.FeatureGroup("Reference route (green)", show=True)
        folium.PolyLine(route_polyline(geo, ref_route), color="#1a9641", weight=6, opacity=0.75).add_to(fg); fg.add_to(m)
    for name, (arcs, col) in (preds or {}).items():
        fg = folium.FeatureGroup(f"{name}", show=True)
        folium.PolyLine(route_polyline(geo, _dedup(arcs)), color=col, weight=3, opacity=0.95).add_to(fg); fg.add_to(m)
    m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])
    legend = f"""<div style='position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 12px;border:1px solid #888;font-size:12px'>
    <b>{title}</b><br><span style='color:#1f5fff'>&#9679;</span> GPS fixes<br><span style='color:#8a8a8a'>&#9632;</span> road network<br>
    <span style='color:#1a9641'>&#9632;</span> reference route<br>""" + "".join(f"<span style='color:{c}'>&#9632;</span> {n}<br>" for n, (a, c) in (preds or {}).items()) + "</div>"
    m.get_root().html.add_child(folium.Element(legend)); folium.LayerControl(collapsed=False).add_to(m)
    return m
def _dedup(a):
    a = np.asarray(a); a = a[a >= 0]
    return a[np.r_[True, a[1:] != a[:-1]]] if len(a) else a
