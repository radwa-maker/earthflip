
import sqlite3, json, time, math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np, pandas as pd, requests
from scipy import stats
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI(title="EARTHFLIP Scientific API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MONITORED_REGIONS = [
    {"id": "ARCTIC_SVALBARD", "name": "Svalbard Archipelago", "biome": "Arctic Tundra", "lat": 78.22, "lon": 15.65},
    {"id": "N_ATLANTIC_HOLE", "name": "Subpolar North Atlantic", "biome": "Marine Subpolar", "lat": 55.00, "lon": -35.00},
    {"id": "MED_CAIRO_DELTA", "name": "Nile Delta & Eastern Med", "biome": "Arid / Mediterranean", "lat": 30.55, "lon": 31.01},
    {"id": "EGYPT_WESTERN_DESERT", "name": "Western Desert (Farafra/Dakhla)", "biome": "Hyper-Arid Desert", "lat": 26.5833, "lon": 28.4167},
    {"id": "INDO_GANGETIC", "name": "Indo-Gangetic Plain (Punjab)", "biome": "Subtropical Agricultural", "lat": 30.90, "lon": 75.85},
    {"id": "AMAZON_CENTRAL", "name": "Central Amazon Basin", "biome": "Tropical Rainforest", "lat": -3.46, "lon": -62.21},
    {"id": "US_MIDWEST", "name": "US Upper Midwest", "biome": "Temperate Continental", "lat": 43.00, "lon": -93.50},
    {"id": "SIBERIA_YAKUTSK", "name": "Central Siberia (Yakutsk)", "biome": "Subarctic Boreal", "lat": 62.03, "lon": 129.73},
    {"id": "SAHEL_NIAMEY", "name": "Central Sahel", "biome": "Semi-Arid Savanna", "lat": 13.51, "lon": 2.11},
    {"id": "SOUTHERN_OCEAN", "name": "Southern Ocean (Drake Passage)", "biome": "Polar Marine", "lat": -60.00, "lon": -65.00},
    {"id": "AUSTRALIA_OUTBACK", "name": "Central Australian Basin", "biome": "Arid Scrubland", "lat": -23.70, "lon": 133.88},
    {"id": "PATAGONIA_SOUTH", "name": "Southern Patagonia", "biome": "Cold Steppe", "lat": -50.33, "lon": -71.25},
]

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "earthflip_nasa_cache.db")
conn = sqlite3.connect(DB_PATH)
conn.execute("CREATE TABLE IF NOT EXISTS nasa_timeseries (region_id TEXT PRIMARY KEY, lat REAL, lon REAL, raw_json TEXT, fetched_at REAL, source_url TEXT)")
conn.commit()
conn.close()

def fetch_nasa_power_monthly(region: Dict[str, Any]) -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT raw_json, source_url FROM nasa_timeseries WHERE region_id = ?", (region["id"],))
    row = cur.fetchone()
    if row:
        conn.close()
        return {"payload": json.loads(row[0]), "source_url": row[1]}
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {"parameters": "T2M", "community": "RE", "longitude": region["lon"], "latitude": region["lat"], "start": 1981, "end": 2025, "format": "JSON"}
    resp = requests.get(url, params=params, timeout=60)
    source_url = resp.url
    resp.raise_for_status()
    payload = resp.json()
    cur.execute("INSERT OR REPLACE INTO nasa_timeseries VALUES (?, ?, ?, ?, ?, ?)", (region["id"], region["lat"], region["lon"], json.dumps(payload), time.time(), source_url))
    conn.commit()
    conn.close()
    return {"payload": payload, "source_url": source_url}

def extract_clean_series(payload: Dict[str, Any], start_year: int, end_year: int) -> pd.DataFrame:
    raw_t2m = payload.get("properties", {}).get("parameter", {}).get("T2M", {})
    records = []
    for ym_key, val in raw_t2m.items():
        if len(ym_key) != 6: continue
        year, month = int(ym_key[:4]), int(ym_key[4:6])
        if month < 1 or month > 12 or year < start_year or year > end_year: continue
        clean_val = np.nan if (val is None or float(val) <= -900.0) else float(val)
        records.append({"year": year, "month": month, "t2m": clean_val})
    return pd.DataFrame(records, columns=["year", "month", "t2m"]).sort_values(["year", "month"]).reset_index(drop=True)

def seasonal_mann_kendall_and_sen(df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Any]:
    expected = len(df)
    valid_df = df.dropna(subset=["t2m"]).copy()
    n_valid = len(valid_df)
    comp = (n_valid / expected) if expected > 0 else 0.0
    if n_valid < 120 or comp < 0.80:
        return {"sufficient": False, "n_valid": n_valid, "n_expected": expected, "completeness_pct": round(comp * 100, 1), "slope_decade": None, "ci_low_decade": None, "ci_high_decade": None, "ols_slope_decade": None, "p_value": None, "autocorr_lag1": None, "limitations": "Insufficient valid monthly observations (< 120 months or < 80% coverage)."}
    S_total, var_S_total, slopes = 0.0, 0.0, []
    for m in range(1, 13):
        sub = valid_df[valid_df["month"] == m].sort_values("year")
        y, t, n_m = sub["t2m"].values, sub["year"].values, len(sub)
        if n_m < 4: continue
        for i in range(n_m - 1):
            diffs = y[i+1:] - y[i]
            S_total += int(np.sum(np.sign(diffs)))
            slopes.extend((diffs / (t[i+1:] - t[i])).tolist())
        _, counts = np.unique(y, return_counts=True)
        var_S_total += (n_m * (n_m - 1) * (2 * n_m + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    anomalies = (valid_df["t2m"] - valid_df.groupby("month")["t2m"].transform("mean")).values
    r1 = float(np.corrcoef(anomalies[:-1], anomalies[1:])[0, 1]) if len(anomalies) > 10 and np.std(anomalies) > 0 else 0.0
    vif = min(max(1.0, 1.0 + 2.0 * (r1 ** 2)), 4.0) if r1 > 0.05 else 1.0
    var_corr = var_S_total * vif
    z = (S_total - 1.0) / math.sqrt(var_corr) if S_total > 0 else ((S_total + 1.0) / math.sqrt(var_corr) if S_total < 0 else 0.0)
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    s_arr = np.sort(np.array(slopes)) * 10.0
    sen_slope = float(np.median(s_arr))
    c_alpha = stats.norm.ppf(1.0 - alpha / 2.0) * math.sqrt(var_corr)
    ci_low = float(s_arr[max(0, int(round((len(s_arr) - c_alpha) / 2.0)))])
    ci_high = float(s_arr[min(len(s_arr) - 1, int(round((len(s_arr) + c_alpha) / 2.0)))])
    ols = stats.linregress(valid_df["year"].values + (valid_df["month"].values - 0.5) / 12.0, anomalies)
    return {"sufficient": True, "n_valid": n_valid, "n_expected": expected, "completeness_pct": round(comp * 100, 1), "slope_decade": round(sen_slope, 4), "ci_low_decade": round(ci_low, 4), "ci_high_decade": round(ci_high, 4), "ols_slope_decade": round(float(ols.slope * 10.0), 4), "p_value": round(p_val, 6), "autocorr_lag1": round(r1, 3), "limitations": "MERRA-2 ~50 km grid smooths local microclimates. Statistical significance does not establish physical causation."}

def apply_benjamini_hochberg(results: List[Dict[str, Any]], fdr_alpha: float = 0.05) -> List[Dict[str, Any]]:
    valid = [i for i, r in enumerate(results) if r["stats"]["sufficient"]]
    m = len(valid)
    p_vals = sorted([(idx, results[idx]["stats"]["p_value"]) for idx in valid], key=lambda x: x[1])
    prev_q = 1.0
    for rank_idx in range(m - 1, -1, -1):
        orig_idx, p = p_vals[rank_idx]
        q = min(prev_q, p * m / (rank_idx + 1))
        prev_q = q
        results[orig_idx]["stats"]["q_value_fdr"] = round(q, 6)
        slope = results[orig_idx]["stats"]["slope_decade"]
        results[orig_idx]["classification"] = "Increasing" if (q < fdr_alpha and slope > 0) else ("Decreasing" if (q < fdr_alpha and slope < 0) else "No Statistically Significant Trend")
    for r in results:
        if not r["stats"]["sufficient"]:
            r["stats"]["q_value_fdr"] = None
            r["classification"] = "Insufficient Data"
    return results

@app.get("/api/analyze")
def analyze_trends(start_year: int = Query(1981), end_year: int = Query(2025), fdr_alpha: float = Query(0.05)):
    if not (1981 <= start_year <= end_year <= 2025): raise HTTPException(status_code=400, detail="Years must be between 1981 and 2025.")
    if end_year - start_year < 9: raise HTTPException(status_code=400, detail="Select at least 10 years.")
    if not (0 < fdr_alpha < 1): raise HTTPException(status_code=400, detail="Invalid FDR alpha.")
    out = []
    for reg in MONITORED_REGIONS:
        try:
            fetched = fetch_nasa_power_monthly(reg)
        except (requests.RequestException, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"NASA data request failed for {reg['name']}: {exc}")
        df = extract_clean_series(fetched["payload"], start_year, end_year)
        st = seasonal_mann_kendall_and_sen(df)
        df["anomaly"] = df["t2m"] - df.groupby("month")["t2m"].transform("mean")
        ann = df.groupby("year", as_index=False).agg(mean_anomaly=("anomaly", "mean"), valid=("t2m", "count"))
        series = [{"year": int(r["year"]), "anomaly": round(float(r["mean_anomaly"]), 3) if r["valid"] >= 10 else None} for _, r in ann.iterrows()]
        out.append({"region": reg, "analysis_period": f"{start_year}-{end_year}", "provenance": {"source_url": fetched["source_url"]}, "stats": st, "annual_series": series})
    out = apply_benjamini_hochberg(out, fdr_alpha)
    inc = [r for r in out if r["classification"] == "Increasing"]
    dec = [r for r in out if r["classification"] == "Decreasing"]
    pairs = [{"increasing_region": i["region"]["name"], "increasing_id": i["region"]["id"], "increasing_slope": i["stats"]["slope_decade"], "increasing_q": i["stats"]["q_value_fdr"], "decreasing_region": d["region"]["name"], "decreasing_id": d["region"]["id"], "decreasing_slope": d["stats"]["slope_decade"], "decreasing_q": d["stats"]["q_value_fdr"], "divergence_magnitude_decade": round(i["stats"]["slope_decade"] - d["stats"]["slope_decade"], 4)} for i in inc for d in dec]
    return {"analysis_period": f"{start_year}-{end_year}", "fdr_alpha": fdr_alpha, "summary_counts": {"total_regions": len(out), "increasing": len(inc), "decreasing": len(dec), "no_significant_trend": len([r for r in out if r["classification"] == "No Statistically Significant Trend"]), "verified_opposite_pairs": len(pairs)}, "opposite_pairs": pairs, "regions": out}

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open(BASE_DIR / "index.html", "r", encoding="utf-8-sig") as f: return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 
