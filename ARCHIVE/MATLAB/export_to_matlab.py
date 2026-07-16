#!/usr/bin/env python3
"""
Bundle every project dataset -- floor grid, materials, georeference,
transmitters, STEP_2 simulation outputs, and the walk-test measurements --
into one MATLAB file (indoor_walk_test.mat), all normalized to shared
coordinate frames:

  pixel frame   : float (x_px, y_px), y down, 0-based (MATLAB: index +1)
  local meters  : ENU about origin_lonlat (use for physical distances)
  lon/lat + EPSG:3857 for joining external data

Walk-test rows keep only samples with a power reading; LTE (Ref Signal) and
NR (SSB) rows are merged into unified rsrp/rsrq/cinr columns with a protocol
flag. GPS drifts far beyond the floor plate indoors, so each point carries an
on_floor flag instead of being dropped.

usage: python MATLAB/export_to_matlab.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[2]        # repo root (file lives in ARCHIVE/MATLAB)
S1, S2 = ROOT / "STEP_1", ROOT / "STEP_2"
WALK_CSV = ROOT / "ARCHIVE" / "raw_walk_data" / "Concat_Indoor_Walk_Test_from_csv.csv"


def invert_affine(M):
    A, b = np.asarray(M)[:, :2], np.asarray(M)[:, 2]
    Ai = np.linalg.inv(A)
    return Ai, -Ai @ b


def main():
    meta = json.loads((S1 / "floorplan_meta.json").read_text())
    mats = json.loads((S1 / "materials.json").read_text())
    txs = json.loads((S1 / "transmitters.json").read_text())["transmitters"]
    sim_params = json.loads((S2 / "sim_params.json").read_text())

    ids = sorted(int(k) for k in mats)
    floor = dict(
        material_grid=np.load(S1 / "material_grid.npy"),
        material_grid_consolidated=np.load(S2 / "material_grid_consolidated.npy"),
        outside_mask=np.load(S2 / "outside_mask.npy"),
        material_ids=np.array(ids, np.int32),
        material_names=np.array([mats[str(i)]["name"] for i in ids], object),
        loss_db=np.array([mats[str(i)]["loss_db"] for i in ids]),
        loss_per_m_db=np.array([mats[str(i)].get("loss_per_m_db", 0.0) for i in ids]),
        meters_per_px=meta["meters_per_px"],
        map_rotation_deg=meta["map_rotation_deg"],
        origin_lonlat=np.array(meta["origin_lonlat"]),
        affine_px_to_local_m=np.array(meta["affine_px_to_local_m"]),
        affine_px_to_mercator=np.array(meta["affine_px_to_mercator"]),
        affine_px_to_lonlat=np.array(meta["affine_px_to_lonlat"]),
        gcp_residuals_m=np.array(meta["gcp_residuals_m"]),
        note="pixel coords are 0-based floats, y down: MATLAB index = "
             "grid(round(y_px)+1, round(x_px)+1). Affines map (x_px, y_px, 1).",
    )

    tx = {k: np.array([t[k] for t in txs]) for k in
          ("x_px", "y_px", "x_m", "y_m", "lon", "lat", "mercator_x", "mercator_y")}

    sim = dict(
        pathloss_tx1_db=np.load(S2 / "pathloss_tx1.npy"),
        pathloss_tx2_db=np.load(S2 / "pathloss_tx2.npy"),
        prx_best_server_dbm=np.load(S2 / "prx_best_server.npy"),
        params_json=json.dumps(sim_params),
        freq_mhz=sim_params["freq_mhz"], n_exp=sim_params["n_exp"],
        eirp_dbm=sim_params["eirp_dbm"], wall_sat_db=sim_params["wall_sat_db"],
    )

    # ---- walk-test measurements, normalized to the floor frames ------------
    df = pd.read_csv(WALK_CSV, low_memory=False)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    lte_rsrp, nr_rsrp = num("Ref Signal - Received Power"), num("SSB - Received Power")
    protocol = np.where(nr_rsrp.notna(), 2, np.where(lte_rsrp.notna(), 1, 0))
    keep = protocol > 0

    rsrp = lte_rsrp.combine_first(nr_rsrp)
    rsrq = num("Ref Signal - Received Quality").combine_first(
        num("SSB - Received Quality"))
    cinr = num("Ref Signal - CINR").combine_first(num("SSB - CINR"))

    lon, lat = num("Longitude")[keep].values, num("Latitude")[keep].values
    Ai, bi = invert_affine(meta["affine_px_to_lonlat"])
    px = (Ai @ np.vstack([lon, lat]) + bi[:, None])
    Am, bm = np.asarray(meta["affine_px_to_local_m"])[:, :2], \
        np.asarray(meta["affine_px_to_local_m"])[:, 2]
    xym = Am @ px + bm[:, None]
    H, W = floor["material_grid"].shape
    on_floor = (px[0] >= 0) & (px[0] < W) & (px[1] >= 0) & (px[1] < H)

    walk = dict(
        datetime=np.array((df["Date"][keep] + " " + df["Time"][keep]).tolist(),
                          object),
        lat=lat, lon=lon,
        x_px=px[0], y_px=px[1], x_m=xym[0], y_m=xym[1],
        on_floor=on_floor,
        protocol=protocol[keep],
        protocol_key=np.array(["1=LTE (Ref Signal)", "2=NR (SSB)"], object),
        rsrp_dbm=rsrp[keep].values, rsrq_db=rsrq[keep].values,
        cinr_db=cinr[keep].values,
        rssi_dbm=num("Channel RSSI")[keep].values,
        pci=num("Cell Id")[keep].values,
        freq_mhz=num("Channel Frequency")[keep].values,
        band=num("Band")[keep].values,
        note="signals are from OUTDOOR macro donors (not the candidate Tx "
             "pins); GPS drifts indoors, hence on_floor flag",
    )

    out = ROOT / "MATLAB" / "indoor_walk_test.mat"
    savemat(out, dict(floorplan=floor, tx=tx, sim=sim, walk=walk),
            do_compression=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"walk rows kept {keep.sum()} of {len(df)} "
          f"(LTE {(protocol == 1).sum()}, NR {(protocol == 2).sum()}), "
          f"on-floor {int(on_floor.sum())}")


if __name__ == "__main__":
    main()
