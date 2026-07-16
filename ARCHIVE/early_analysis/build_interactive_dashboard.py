from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "CSV"
FLOOR_MAP_PNG = ROOT / "7th_Floor_2nd_Indoor_Walk_Test_V2.2.png"
FLOOR_MAP_TAB = ROOT / "7th_Floor_2nd_Indoor_Walk_Test_V2.2.TAB"
OUTPUT_HTML = ROOT / "Frontend_Data_Display.html"

PAT_GENERIC_NR_TOPN = "*nr Top N Signal*.CSV"
METRICS = ("rsrp", "rsrq", "cinr", "rssi")
UNITS = {
    "rsrp": "dBm",
    "rsrq": "dB",
    "cinr": "dB",
    "rssi": "dBm",
}
SIGNAL_RANGES = {
    "rsrp": (-140.0, -44.0),
    "rsrq": (-43.0, 20.0),
    "cinr": (-23.0, 40.0),
    "rssi": (-100.0, 0.0),
}


def read_tab_gcps(tab_path: Path) -> pd.DataFrame:
    pattern = re.compile(r"\((-?[\d.]+),(-?[\d.]+)\)\s*\((\d+),(\d+)\)")
    gcps = []
    for line in tab_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            lon, lat, px, py = map(float, match.groups())
            gcps.append((lon, lat, px, py))
    if len(gcps) < 3:
        raise ValueError("At least 3 GCP points are required in TAB file for affine fit.")
    return pd.DataFrame(gcps, columns=["longitude", "latitude", "px", "py"])


def fit_affine(gcps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    A = np.column_stack([gcps["longitude"], gcps["latitude"], np.ones(len(gcps))])
    coef_x, *_ = np.linalg.lstsq(A, gcps["px"], rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, gcps["py"], rcond=None)

    res_x = A @ coef_x - gcps["px"]
    res_y = A @ coef_y - gcps["py"]
    rmse = float(np.sqrt(np.mean(res_x**2 + res_y**2)))
    return coef_x, coef_y, rmse


def add_pixel_coords(df: pd.DataFrame, coef_x: np.ndarray, coef_y: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    A = np.column_stack([out["longitude"], out["latitude"], np.ones(len(out))])
    out["px"] = A @ coef_x
    out["py"] = A @ coef_y
    return out


def infer_network(source_value: str) -> str:
    upper = str(source_value).upper()
    if "NR_FR2" in upper:
        return "nr_fr2"
    if "NR_FR1" in upper:
        return "nr_fr1"
    return "nr_unknown"


def prepare_nr_topn_metrics_dataframe(csv_dir: Path, pattern: str = PAT_GENERIC_NR_TOPN) -> pd.DataFrame:
    frames = []
    for path in sorted(csv_dir.glob(pattern)):
        if "_BestServing" in path.name:
            continue
        df = pd.read_csv(path)
        df["source"] = path.stem
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "latitude",
                "longitude",
                "pci",
                "freq",
                "band",
                "network",
                "rsrp",
                "rsrq",
                "cinr",
                "rssi",
                "source",
            ]
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Cell ID": "pci",
            "Channel Frequency": "freq",
            "Band": "band",
            "SSS_RP": "rsrp",
            "SSS_RQ": "rsrq",
            "SS_CINR": "cinr",
            "SSB RSSI": "rssi",
        }
    )

    for metric in ["latitude", "longitude", "pci", "freq", "band", "rsrp", "rsrq", "cinr", "rssi"]:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df["network"] = df["source"].map(infer_network)
    columns = ["latitude", "longitude", "pci", "freq", "band", "network", "rsrp", "rsrq", "cinr", "rssi", "source"]
    return df[columns].dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def average_metrics_by_coordinate(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["rsrp", "rsrq", "cinr", "rssi"]
    aggregations = {metric: "mean" for metric in metric_columns}
    aggregations["source"] = lambda values: "; ".join(sorted(set(map(str, values))))
    aggregations["pci"] = "first"
    aggregations["freq"] = "first"

    grouped = df.groupby(["network", "band", "latitude", "longitude"], as_index=False).agg(aggregations)
    grouped["n_samples"] = df.groupby(["network", "band", "latitude", "longitude"]).size().values
    return grouped


def metric_range(values: pd.Series, metric: str, q_low: float = 0.10, q_high: float = 0.90) -> tuple[float, float]:
    clean = values.dropna()
    if clean.empty:
        return SIGNAL_RANGES[metric]

    vmin = float(clean.quantile(q_low))
    vmax = float(clean.quantile(q_high))
    hard_min, hard_max = SIGNAL_RANGES[metric]
    vmin = max(vmin, hard_min)
    vmax = min(vmax, hard_max)
    if vmin >= vmax:
        vmin = float(clean.min())
        vmax = float(clean.max())
    return vmin, vmax


def image_as_data_uri(path: Path) -> str:
    mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_dashboard_html(
    records: list[dict],
    image_data_uri: str,
    image_width: int,
    image_height: int,
    ranges: dict[str, dict[str, float]],
    rmse_px: float,
) -> str:
    records_json = json.dumps(records, ensure_ascii=True)
    ranges_json = json.dumps(ranges, ensure_ascii=True)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Indoor Walk Test Interactive Dashboard</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {{
      --bg: #f4f6f3;
      --panel: #ffffff;
      --ink: #1f2a2a;
      --muted: #5b6662;
      --accent: #0a7f7a;
      --line: #d9e0dc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Helvetica Neue", Helvetica, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 8%, #ddebe6 0%, transparent 35%),
        radial-gradient(circle at 85% 90%, #d7e4f0 0%, transparent 33%),
        var(--bg);
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      gap: 14px;
      grid-template-columns: 300px 1fr;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 22px rgba(24, 40, 36, 0.06);
      padding: 16px;
    }}
    .title {{ margin: 0 0 8px 0; font-size: 1.2rem; }}
    .sub {{ margin: 0 0 14px 0; color: var(--muted); font-size: 0.92rem; }}
    .field {{ margin-bottom: 12px; }}
    .field label {{ display: block; font-size: 0.84rem; color: var(--muted); margin-bottom: 6px; }}
    .field select, .field input {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 0.94rem;
      background: #fff;
      color: var(--ink);
    }}
    .stats {{
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      display: grid;
      gap: 5px;
      font-size: 0.9rem;
    }}
    .pill {{
      display: inline-block;
      margin-top: 8px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #e7f4f2;
      color: var(--accent);
      font-weight: 600;
      font-size: 0.8rem;
    }}
    #plotPanel {{ padding: 10px; }}
    #plot {{ width: 100%; min-height: 760px; }}
    .fade-in {{ animation: fade 420ms ease-out both; }}
    @keyframes fade {{
      from {{ opacity: 0; transform: translateY(4px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (max-width: 1100px) {{
      .wrap {{ grid-template-columns: 1fr; }}
      #plot {{ min-height: 620px; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\"> 
    <aside class=\"panel fade-in\"> 
      <h1 class=\"title\">Indoor Walk Test Viewer</h1>
      <p class=\"sub\">Interactive NR Top-N signal map with network and band filtering.</p>

      <div class=\"field\">
        <label for=\"coverage\">Coverage Mode</label>
        <select id=\"coverage\">
          <option value=\"points\">Points</option>
          <option value=\"gradient\">Gradient Coverage</option>
        </select>
      </div>

      <div class=\"field\">
        <label for=\"network\">Network</label>
        <select id=\"network\"></select>
      </div>

      <div class=\"field\">
        <label for=\"band\">Band</label>
        <select id=\"band\"></select>
      </div>

      <div class=\"field\">
        <label for=\"metric\">Metric</label>
        <select id=\"metric\">
          <option value=\"rsrp\">RSRP (dBm)</option>
          <option value=\"rsrq\">RSRQ (dB)</option>
          <option value=\"cinr\">CINR (dB)</option>
          <option value=\"rssi\">RSSI (dBm)</option>
        </select>
      </div>

      <div class=\"field\">
        <label for=\"vmin\">Color Min</label>
        <input id=\"vmin\" type=\"number\" step=\"0.1\" />
      </div>

      <div class=\"field\">
        <label for=\"vmax\">Color Max</label>
        <input id=\"vmax\" type=\"number\" step=\"0.1\" />
      </div>

      <div class=\"field\">
        <label for=\"size\">Marker Size</label>
        <input id=\"size\" type=\"range\" min=\"5\" max=\"35\" value=\"14\" />
      </div>

      <div class=\"stats\" id=\"stats\"></div>
      <span class=\"pill\">Affine RMSE: {rmse_px:.2f}px</span>
    </aside>

    <section id=\"plotPanel\" class=\"panel fade-in\">
      <div id=\"plot\"></div>
    </section>
  </div>

  <script>
    const records = {records_json};
    const ranges = {ranges_json};
    const units = {{ rsrp: 'dBm', rsrq: 'dB', cinr: 'dB', rssi: 'dBm' }};
    const plotEl = document.getElementById('plot');
    const metricEl = document.getElementById('metric');
    const coverageEl = document.getElementById('coverage');
    const networkEl = document.getElementById('network');
    const bandEl = document.getElementById('band');
    const vminEl = document.getElementById('vmin');
    const vmaxEl = document.getElementById('vmax');
    const sizeEl = document.getElementById('size');
    const statsEl = document.getElementById('stats');

    function uniqueSorted(values) {{
      return [...new Set(values)].sort((a, b) => String(a).localeCompare(String(b), undefined, {{ numeric: true }}));
    }}

    function networkOptions() {{
      return uniqueSorted(records.map(r => r.network).filter(v => typeof v === 'string' && v.length));
    }}

    function bandOptions(networkValue) {{
      const scoped = records.filter(r => networkValue === 'all' || r.network === networkValue);
      return uniqueSorted(scoped.map(r => r.band).filter(v => Number.isFinite(v)));
    }}

    function populateNetworks() {{
      const current = networkEl.value || 'all';
      networkEl.innerHTML = '';

      const allOpt = document.createElement('option');
      allOpt.value = 'all';
      allOpt.textContent = 'All';
      networkEl.appendChild(allOpt);

      for (const net of networkOptions()) {{
        const opt = document.createElement('option');
        opt.value = net;
        opt.textContent = net;
        networkEl.appendChild(opt);
      }}

      networkEl.value = [...networkEl.options].some(o => o.value === current) ? current : 'all';
    }}

    function populateBands() {{
      const current = bandEl.value || 'all';
      bandEl.innerHTML = '';

      const allOpt = document.createElement('option');
      allOpt.value = 'all';
      allOpt.textContent = 'All';
      bandEl.appendChild(allOpt);

      for (const b of bandOptions(networkEl.value || 'all')) {{
        const opt = document.createElement('option');
        opt.value = String(b);
        opt.textContent = `Band ${{b}}`;
        bandEl.appendChild(opt);
      }}

      bandEl.value = [...bandEl.options].some(o => o.value === current) ? current : 'all';
    }}

    function filtered(metric) {{
      const selectedNetwork = networkEl.value || 'all';
      const selectedBand = bandEl.value || 'all';
      return records.filter(r =>
        Number.isFinite(r.px) &&
        Number.isFinite(r.py) &&
        Number.isFinite(r[metric]) &&
        (selectedNetwork === 'all' || r.network === selectedNetwork) &&
        (selectedBand === 'all' || String(r.band) === selectedBand)
      );
    }}

    function metricValues(data, metric) {{
      return data.map(r => r[metric]).filter(v => Number.isFinite(v));
    }}

    function quantile(values, q) {{
      if (!values.length) return NaN;
      const sorted = [...values].sort((a, b) => a - b);
      const pos = (sorted.length - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      if (sorted[base + 1] !== undefined) {{
        return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
      }}
      return sorted[base];
    }}

    function computeDefaultRange(data, metric) {{
      const vals = metricValues(data, metric);
      if (!vals.length) {{
        return ranges[metric];
      }}
      const q10 = quantile(vals, 0.10);
      const q90 = quantile(vals, 0.90);
      const hard = ranges[metric];
      let vmin = Math.max(q10, hard.vmin);
      let vmax = Math.min(q90, hard.vmax);
      if (!(vmin < vmax)) {{
        vmin = Math.min(...vals);
        vmax = Math.max(...vals);
      }}
      return {{ vmin, vmax }};
    }}

    function idwGrid(data, metric, width, height, nx = 110, ny = 80, power = 2.0) {{
      if (data.length < 3) return null;
      const xVals = Array.from({{ length: nx }}, (_, i) => (i * width) / (nx - 1));
      const yVals = Array.from({{ length: ny }}, (_, i) => (i * height) / (ny - 1));
      const z = [];

      for (let yi = 0; yi < ny; yi++) {{
        const row = [];
        const y = yVals[yi];
        for (let xi = 0; xi < nx; xi++) {{
          const x = xVals[xi];
          let num = 0;
          let den = 0;
          let exact = null;

          for (const p of data) {{
            const v = p[metric];
            if (!Number.isFinite(v)) continue;
            const dx = x - p.px;
            const dy = y - p.py;
            const d2 = dx * dx + dy * dy;
            if (d2 < 1e-8) {{
              exact = v;
              break;
            }}
            const w = 1.0 / Math.pow(d2, power / 2.0);
            num += w * v;
            den += w;
          }}

          if (exact !== null) {{
            row.push(exact);
          }} else if (den > 0) {{
            row.push(num / den);
          }} else {{
            row.push(null);
          }}
        }}
        z.push(row);
      }}

      return {{ x: xVals, y: yVals, z }};
    }}

    function hoverTemplate(metric) {{
      const u = units[metric] || '';
      return [
        `<b>${{metric.toUpperCase()}}</b>: %{{{{marker.color:.2f}}}} ${{u}}`,
        'Lat: %{{customdata[0]:.6f}}',
        'Lon: %{{customdata[1]:.6f}}',
        'Samples: %{{customdata[2]}}',
        'PCI: %{{customdata[3]}}',
        'Freq: %{{customdata[4]}}',
        '<extra></extra>'
      ].join('<br>');
    }}

    function updateStats(data, metric) {{
      const vals = metricValues(data, metric);
      const n = vals.length;
      const min = n ? Math.min(...vals).toFixed(2) : 'NA';
      const max = n ? Math.max(...vals).toFixed(2) : 'NA';
      const avg = n ? (vals.reduce((a,b) => a+b, 0) / n).toFixed(2) : 'NA';
      const networkText = networkEl.value === 'all' ? 'All' : networkEl.value;
      const bandText = bandEl.value === 'all' ? 'All' : `Band ${{bandEl.value}}`;
      const modeText = coverageEl.value;
      statsEl.innerHTML = `
        <div>Points: <b>${{n}}</b></div>
        <div>Metric: <b>${{metric.toUpperCase()}}</b></div>
        <div>Mode: <b>${{modeText}}</b></div>
        <div>Network / Band: <b>${{networkText}} / ${{bandText}}</b></div>
        <div>Min / Max: <b>${{min}} / ${{max}}</b></div>
        <div>Mean: <b>${{avg}}</b></div>
      `;
    }}

    function applyDefaults(metric, data) {{
      const d = computeDefaultRange(data, metric);
      vminEl.value = d.vmin.toFixed(2);
      vmaxEl.value = d.vmax.toFixed(2);
    }}

    function render() {{
      const metric = metricEl.value;
      const data = filtered(metric);
      const cmin = Number(vminEl.value);
      const cmax = Number(vmaxEl.value);
      const size = Number(sizeEl.value);
      const mode = coverageEl.value;

      const traces = [];

      if (mode === 'gradient') {{
        const grid = idwGrid(data, metric, {image_width}, {image_height});
        if (grid) {{
          traces.push({{
            type: 'heatmap',
            x: grid.x,
            y: grid.y,
            z: grid.z,
            colorscale: 'Viridis',
            zmin: cmin,
            zmax: cmax,
            opacity: 0.72,
            colorbar: {{
              title: `${{metric.toUpperCase()}} (${{units[metric] || ''}})`,
              len: 0.8,
              thickness: 16
            }},
            hovertemplate: `${{metric.toUpperCase()}}: %{{z:.2f}} ${{units[metric] || ''}}<extra></extra>`
          }});
        }}
        traces.push({{
          type: 'scattergl',
          mode: 'markers',
          x: data.map(r => r.px),
          y: data.map(r => r.py),
          customdata: data.map(r => [r.latitude, r.longitude, r.n_samples, r.pci, r.freq]),
          marker: {{ size: Math.max(4, Math.floor(size / 3)), color: '#111', opacity: 0.35 }},
          hovertemplate: hoverTemplate(metric),
          showlegend: false
        }});
      }} else {{
        traces.push({{
          type: 'scattergl',
          mode: 'markers',
          x: data.map(r => r.px),
          y: data.map(r => r.py),
          customdata: data.map(r => [r.latitude, r.longitude, r.n_samples, r.pci, r.freq]),
          marker: {{
            size,
            color: metricValues(data, metric),
            colorscale: 'Viridis',
            cmin,
            cmax,
            colorbar: {{
              title: `${{metric.toUpperCase()}} (${{units[metric] || ''}})`,
              len: 0.8,
              thickness: 16
            }},
            line: {{color: '#111', width: 0.35}},
            opacity: 0.88
          }},
          hovertemplate: hoverTemplate(metric)
        }});
      }}

      const layout = {{
        margin: {{l: 12, r: 16, t: 40, b: 12}},
        title: {{
          text: `${{(networkEl.value === 'all' ? 'All Networks' : networkEl.value)}} · ${{(bandEl.value === 'all' ? 'All Bands' : ('Band ' + bandEl.value))}} · ${{metric.toUpperCase()}}`,
          x: 0.02
        }},
        xaxis: {{visible: false, range: [0, {image_width}] }},
        yaxis: {{visible: false, range: [{image_height}, 0], scaleanchor: 'x'}},
        images: [{{
          source: '{image_data_uri}',
          xref: 'x',
          yref: 'y',
          x: 0,
          y: 0,
          sizex: {image_width},
          sizey: {image_height},
          xanchor: 'left',
          yanchor: 'top',
          sizing: 'stretch',
          layer: 'below'
        }}],
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff'
      }};

      Plotly.react(plotEl, traces, layout, {{responsive: true, displaylogo: false}});
      updateStats(data, metric);
    }}

    metricEl.addEventListener('change', () => {{
      const data = filtered(metricEl.value);
      applyDefaults(metricEl.value, data);
      render();
    }});
    coverageEl.addEventListener('change', render);
    networkEl.addEventListener('change', () => {{
      populateBands();
      const data = filtered(metricEl.value);
      applyDefaults(metricEl.value, data);
      render();
    }});
    bandEl.addEventListener('change', () => {{
      const data = filtered(metricEl.value);
      applyDefaults(metricEl.value, data);
      render();
    }});
    vminEl.addEventListener('change', render);
    vmaxEl.addEventListener('change', render);
    sizeEl.addEventListener('input', render);

    populateNetworks();
    populateBands();
    applyDefaults(metricEl.value, filtered(metricEl.value));
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    if not CSV_DIR.exists():
        raise FileNotFoundError(f"CSV directory not found: {CSV_DIR}")
    if not FLOOR_MAP_PNG.exists():
        raise FileNotFoundError(f"Floor map PNG not found: {FLOOR_MAP_PNG}")
    if not FLOOR_MAP_TAB.exists():
        raise FileNotFoundError(f"Floor map TAB not found: {FLOOR_MAP_TAB}")

    nr_metrics = prepare_nr_topn_metrics_dataframe(CSV_DIR)
    if nr_metrics.empty:
      raise ValueError("No NR Top-N rows found in CSV files.")

    avg_df = average_metrics_by_coordinate(nr_metrics)
    gcps = read_tab_gcps(FLOOR_MAP_TAB)
    coef_x, coef_y, rmse = fit_affine(gcps)
    plot_df = add_pixel_coords(avg_df, coef_x, coef_y)

    records = []
    for row in plot_df.itertuples(index=False):
        records.append(
            {
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "px": float(row.px),
                "py": float(row.py),
                "rsrp": float(row.rsrp) if pd.notna(row.rsrp) else None,
                "rsrq": float(row.rsrq) if pd.notna(row.rsrq) else None,
                "cinr": float(row.cinr) if pd.notna(row.cinr) else None,
                "rssi": float(row.rssi) if pd.notna(row.rssi) else None,
                "n_samples": int(row.n_samples) if pd.notna(row.n_samples) else 0,
                "pci": int(row.pci) if pd.notna(row.pci) else None,
                "freq": float(row.freq) if pd.notna(row.freq) else None,
                "network": str(row.network),
                "band": int(row.band) if pd.notna(row.band) else None,
            }
        )

    metric_ranges = {}
    for metric in METRICS:
        vmin, vmax = metric_range(plot_df[metric], metric)
        metric_ranges[metric] = {
            "vmin": float(vmin),
            "vmax": float(vmax),
            "unit": UNITS[metric],
        }

    # Lightweight PNG header parser for width/height to avoid extra dependencies.
    raw = FLOOR_MAP_PNG.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Floor map image is not a valid PNG file.")
    image_width = int.from_bytes(raw[16:20], "big")
    image_height = int.from_bytes(raw[20:24], "big")

    html = build_dashboard_html(
        records=records,
        image_data_uri=image_as_data_uri(FLOOR_MAP_PNG),
        image_width=image_width,
        image_height=image_height,
        ranges=metric_ranges,
        rmse_px=rmse,
    )
    OUTPUT_HTML.write_text(html, encoding="utf-8")

    print(f"rows_input={len(nr_metrics)}")
    print(f"rows_unique_coordinates={len(plot_df)}")
    print(f"networks={sorted(plot_df['network'].dropna().unique().tolist())}")
    print(f"bands={sorted(plot_df['band'].dropna().astype(int).unique().tolist())}")
    print(f"output_html={OUTPUT_HTML}")


if __name__ == "__main__":
    main()