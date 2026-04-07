import pandas as pd
import numpy as np
import os
from scipy import stats
import pymannkendall as mk

# ==============================
# SETUP
# ==============================

data_path = "./data"

file_city_map = {
    "Bangalore_1990_2022_BangaloreCity.csv": "Bangalore",
    "Chennai_1990_2022_Madras.csv": "Chennai",
    "Delhi_NCR_1990_2022_Safdarjung.csv": "Delhi",
    "Lucknow_1990_2022.csv": "Lucknow",
    "Mumbai_1990_2022_Santacruz.csv": "Mumbai",
    # NOTE: Rajasthan_1990_2022_Jodhpur.csv is a byte-for-byte duplicate of
    # Bangalore_1990_2022_BangaloreCity.csv (confirmed via MD5 hash).
    # This file was mislabeled at source — there is no actual Jodhpur/Rajasthan data.
    # It has been removed to prevent duplicate analysis.
    # TO FIX: replace with a genuine Rajasthan/Jodhpur weather CSV when available.
    "weather_Bhubhneshwar_1990_2022.csv": "Bhubaneswar",
    "weather_Rourkela_2021_2022.csv": "Rourkela"
}

df_list = []
for file, city in file_city_map.items():
    temp_df = pd.read_csv(os.path.join(data_path, file))
    temp_df["city"] = city
    df_list.append(temp_df)

master_df = pd.concat(df_list, ignore_index=True)
master_df["time"] = pd.to_datetime(master_df["time"], format="mixed", errors="coerce")
master_df = master_df.dropna(subset=["time"])
master_df["year"] = master_df["time"].dt.year
master_df["month"] = master_df["time"].dt.month
master_df = master_df.drop(columns=["snow", "wpgt", "tsun", "pres", "wdir", "wspd"])
master_df = master_df.drop_duplicates(subset=["city", "time"])

master_df.loc[
    (master_df["city"] == "Delhi") & (master_df["time"] == "2003-07-06"), "tmin"
] = pd.NA

master_df = master_df.sort_values(["city", "time"]).reset_index(drop=True)

season_map = {
    12: "Winter",       1: "Winter",       2: "Winter",
    3:  "Pre-Monsoon",  4: "Pre-Monsoon",  5: "Pre-Monsoon",
    6:  "Monsoon",      7: "Monsoon",      8: "Monsoon",      9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon"
}
master_df["season"] = master_df["month"].map(season_map)

# ==============================
# COVERAGE FLAGS
# ==============================

coverage_records = []
for city in master_df["city"].unique():
    city_df = master_df[master_df["city"] == city]
    for year in city_df["year"].unique():
        year = int(year)
        year_df = city_df[city_df["year"] == year]
        start = pd.Timestamp(f"{year}-01-01")
        end   = pd.Timestamp(f"{year}-12-31")
        expected_days = (end - start).days + 1
        coverage_records.append({
            "city": city, "year": year,
            "tavg_coverage": year_df["tavg"].notnull().sum() / expected_days * 100,
            "prcp_coverage": year_df["prcp"].notnull().sum() / expected_days * 100,
            "tmax_coverage": year_df["tmax"].notnull().sum() / expected_days * 100,
        })

coverage_df = pd.DataFrame(coverage_records)

monthly_prcp_coverage = (
    master_df.groupby(["city", "year", "month"])["prcp"]
    .apply(lambda x: x.notnull().mean() * 100)
    .reset_index(name="prcp_monthly_coverage")
)
monsoon_min = (
    monthly_prcp_coverage[monthly_prcp_coverage["month"].isin([6, 7, 8, 9])]
    .groupby(["city", "year"])["prcp_monthly_coverage"]
    .min()
    .reset_index(name="monsoon_min_coverage")
)
coverage_df = coverage_df.merge(monsoon_min, on=["city", "year"], how="left")
coverage_df["temp_valid"] = coverage_df["tavg_coverage"] >= 90
coverage_df["prcp_valid"] = (
    (coverage_df["prcp_coverage"] >= 80) &
    (coverage_df["monsoon_min_coverage"] >= 60)
)

MIN_YEARS = 10

# ==============================
# THRESHOLDS  (IMD definitions)
# ==============================

HEATWAVE_TMAX      = 40.0
HEATWAVE_MIN_DAYS  = 3
WARM_NIGHTS_PCTILE = 90
HEAVY_RAIN         = 64.5
VERY_HEAVY_RAIN    = 115.6
EXTREME_RAIN       = 204.4
DROUGHT_DEFICIT    = -20.0

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def print_section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)

def mk_trend_on_annual(annual_series_df, value_col):
    df = annual_series_df.dropna(subset=[value_col]).sort_values("year")
    if len(df) < MIN_YEARS:
        return None, None, None
    result = mk.original_test(df[value_col].values)
    return round(result.slope * 10, 3), round(result.p, 4), result.trend

# ==============================
# 1. HEATWAVE ANALYSIS
# ==============================

print_section("1. HEATWAVE EVENTS  (tmax >= 40C for >= 3 consecutive days, IMD definition)")

def extract_heatwave_events(city_df, tmax_thresh, min_days):
    events = []
    rows = city_df.dropna(subset=["tmax"]).reset_index(drop=True)
    i = 0
    while i < len(rows):
        if rows.loc[i, "tmax"] >= tmax_thresh:
            j = i
            while j < len(rows) and rows.loc[j, "tmax"] >= tmax_thresh:
                if j > i:
                    gap = (rows.loc[j, "time"] - rows.loc[j-1, "time"]).days
                    if gap > 1:
                        break
                j += 1
            duration = j - i
            if duration >= min_days:
                events.append({
                    "start_date":    rows.loc[i, "time"].date(),
                    "end_date":      rows.loc[j-1, "time"].date(),
                    "duration_days": duration,
                    "peak_tmax":     rows.loc[i:j-1, "tmax"].max(),
                    "mean_tmax":     round(rows.loc[i:j-1, "tmax"].mean(), 2),
                    "year":          rows.loc[i, "year"],
                    "month":         rows.loc[i, "month"],
                })
            i = j
        else:
            i += 1
    return events

all_hw_events = []
hw_summary    = []

for city in sorted(master_df["city"].unique()):
    city_df = master_df[master_df["city"] == city].reset_index(drop=True)
    events  = extract_heatwave_events(city_df, HEATWAVE_TMAX, HEATWAVE_MIN_DAYS)
    for e in events:
        all_hw_events.append({"city": city, **e})
    n_events   = len(events)
    total_days = sum(e["duration_days"] for e in events)
    mean_dur   = round(np.mean([e["duration_days"] for e in events]), 1) if events else 0
    max_dur    = max((e["duration_days"] for e in events), default=0)
    peak_tmax  = max((e["peak_tmax"]    for e in events), default=np.nan)
    hw_summary.append({
        "city": city, "total_events": n_events,
        "total_days": total_days, "mean_duration": mean_dur,
        "max_duration": max_dur, "peak_tmax": peak_tmax
    })
    print(f"  {city:<15} events={n_events:>3}  total_days={total_days:>4}  "
          f"mean_dur={mean_dur:>4}d  peak={peak_tmax}C")

hw_events_df  = pd.DataFrame(all_hw_events)
hw_summary_df = pd.DataFrame(hw_summary)

if not hw_events_df.empty:
    annual_hw = (
        hw_events_df.groupby(["city", "year"])["duration_days"]
        .sum()
        .reset_index(name="hw_days")
    )
    print("\n  Top 10 longest individual heatwave events:")
    top_hw = (
        hw_events_df.sort_values("duration_days", ascending=False)
        .head(10)[["city", "start_date", "end_date", "duration_days", "peak_tmax"]]
    )
    print(top_hw.to_string(index=False))

    print("\n  Trend in annual heatwave days (Sen's slope, days/decade):")
    for city in sorted(annual_hw["city"].unique()):
        c_df = annual_hw[annual_hw["city"] == city]
        all_years = pd.DataFrame({"year": range(int(c_df["year"].min()),
                                                 int(c_df["year"].max()) + 1)})
        c_df = all_years.merge(c_df, on="year", how="left").fillna(0)
        slope, p, direction = mk_trend_on_annual(c_df, "hw_days")
        if slope is not None:
            print(f"    {city:<15} {slope:+.2f} days/dec  p={p:.4f} {sig_stars(p)}")

# ==============================
# 2. WARM NIGHTS (ETCCDI TN90p)
# ==============================

print_section("2. WARM NIGHTS  (tmin > 90th percentile of city-month historical tmin)")
print("  ETCCDI TN90p index. Counts nights too warm for the body to recover.")
print("  Directly linked to Lucknow tmin surge found in script 04.\n")

tmin_pctile = (
    master_df.groupby(["city", "month"])["tmin"]
    .quantile(WARM_NIGHTS_PCTILE / 100)
    .rename("tmin_p90")
    .reset_index()
)
master_df = master_df.merge(tmin_pctile, on=["city", "month"], how="left")
master_df["warm_night"] = master_df["tmin"] > master_df["tmin_p90"]

annual_warm_nights = (
    master_df.groupby(["city", "year"])["warm_night"]
    .sum()
    .reset_index(name="warm_nights")
    .merge(coverage_df[["city", "year", "temp_valid"]], on=["city", "year"], how="left")
)

print(f"  {'City':<15} {'Mean warm nights/yr':>20}  Trend (days/dec)  p-value")
print(f"  {'-'*62}")
for city in sorted(annual_warm_nights["city"].unique()):
    valid    = annual_warm_nights[
        (annual_warm_nights["city"] == city) &
        (annual_warm_nights["temp_valid"] == True)
    ]
    mean_wn  = valid["warm_nights"].mean()
    slope, p, _ = mk_trend_on_annual(valid, "warm_nights")
    if slope is not None:
        print(f"  {city:<15} {mean_wn:>20.1f}  {slope:>+14.2f}  p={p:.4f} {sig_stars(p)}")
    else:
        print(f"  {city:<15} {mean_wn:>20.1f}  {'insufficient data':>16}")

# ==============================
# 3. EXTREME RAINFALL
# ==============================

print_section("3. EXTREME RAINFALL EVENTS  (IMD daily classification)")

master_df["heavy_day"]      = master_df["prcp"] >= HEAVY_RAIN
master_df["very_heavy_day"] = master_df["prcp"] >= VERY_HEAVY_RAIN
master_df["extreme_day"]    = master_df["prcp"] >= EXTREME_RAIN

print(f"\n  Total extreme rain days per city (1990-2022):\n")
print(f"  {'City':<15} {'Heavy(>64mm)':>14} {'VeryHeavy(>115mm)':>18} {'Extreme(>204mm)':>16}")
print(f"  {'-'*65}")
for city in sorted(master_df["city"].unique()):
    c  = master_df[master_df["city"] == city]
    print(f"  {city:<15} {c['heavy_day'].sum():>14} {c['very_heavy_day'].sum():>18} "
          f"{c['extreme_day'].sum():>16}")

print("\n  Top 10 wettest single days on record:")
top_rain = (
    master_df[master_df["prcp"].notnull()]
    .nlargest(10, "prcp")[["city", "time", "prcp"]]
    .rename(columns={"prcp": "rainfall_mm"})
)
print(top_rain.to_string(index=False))

annual_rain_extremes = (
    master_df.groupby(["city", "year"])
    .agg(
        heavy_days      =("heavy_day",      "sum"),
        very_heavy_days =("very_heavy_day", "sum"),
        extreme_days    =("extreme_day",    "sum"),
        max_daily_prcp  =("prcp",           "max")
    )
    .reset_index()
    .merge(coverage_df[["city", "year", "prcp_valid"]], on=["city", "year"], how="left")
)

print("\n  Trend in annual heavy rain days (Sen's slope, days/decade):")
for city in sorted(annual_rain_extremes["city"].unique()):
    valid      = annual_rain_extremes[
        (annual_rain_extremes["city"] == city) &
        (annual_rain_extremes["prcp_valid"] == True)
    ]
    slope, p, _ = mk_trend_on_annual(valid, "heavy_days")
    mean_heavy  = valid["heavy_days"].mean()
    if slope is not None:
        print(f"    {city:<15} mean={mean_heavy:.1f}/yr  trend={slope:+.2f} days/dec  "
              f"p={p:.4f} {sig_stars(p)}")
    else:
        print(f"    {city:<15} insufficient valid years")

# ==============================
# 4. CONSECUTIVE WET DAYS (CWD)
# ==============================

print_section("4. CONSECUTIVE WET DAYS  (CWD - ETCCDI index)")
print("  Max consecutive days with prcp >= 1mm per year.")
print("  Long CWD = sustained monsoon. Short CWD = fragmented rainfall.\n")

def max_consecutive_wet(series, threshold=1.0):
    wet     = (series.fillna(0) >= threshold)
    max_run = 0
    current = 0
    for v in wet:
        current = current + 1 if v else 0
        max_run = max(max_run, current)
    return max_run

cwd_records = []
for city in sorted(master_df["city"].unique()):
    city_df = master_df[master_df["city"] == city]
    for year, yr_df in city_df.groupby("year"):
        cwd_records.append({"city": city, "year": year,
                            "cwd": max_consecutive_wet(yr_df["prcp"])})

cwd_df = (pd.DataFrame(cwd_records)
          .merge(coverage_df[["city", "year", "prcp_valid"]], on=["city", "year"], how="left"))

print(f"  {'City':<15} {'Mean CWD':>10}  Trend (days/dec)  p-value")
print(f"  {'-'*55}")
for city in sorted(cwd_df["city"].unique()):
    valid    = cwd_df[(cwd_df["city"] == city) & (cwd_df["prcp_valid"] == True)]
    mean_cwd = valid["cwd"].mean()
    slope, p, _ = mk_trend_on_annual(valid, "cwd")
    if slope is not None:
        print(f"  {city:<15} {mean_cwd:>10.1f}  {slope:>+14.2f}  p={p:.4f} {sig_stars(p)}")
    else:
        print(f"  {city:<15} {mean_cwd:>10.1f}  insufficient data")

# ==============================
# 5. CONSECUTIVE DRY DAYS (CDD)
# ==============================

print_section("5. CONSECUTIVE DRY DAYS  (CDD - ETCCDI index)")
print("  Max consecutive days with prcp < 1mm per year.")
print("  Rising CDD = longer dry spells = groundwater and crop stress.\n")

def max_consecutive_dry(series, threshold=1.0):
    dry     = (series.fillna(0) < threshold)
    max_run = 0
    current = 0
    for v in dry:
        current = current + 1 if v else 0
        max_run = max(max_run, current)
    return max_run

cdd_records = []
for city in sorted(master_df["city"].unique()):
    city_df = master_df[master_df["city"] == city]
    for year, yr_df in city_df.groupby("year"):
        cdd_records.append({"city": city, "year": year,
                            "cdd": max_consecutive_dry(yr_df["prcp"])})

cdd_df = (pd.DataFrame(cdd_records)
          .merge(coverage_df[["city", "year", "prcp_valid"]], on=["city", "year"], how="left"))

print(f"  {'City':<15} {'Mean CDD':>10}  Trend (days/dec)  p-value")
print(f"  {'-'*55}")
for city in sorted(cdd_df["city"].unique()):
    valid    = cdd_df[(cdd_df["city"] == city) & (cdd_df["prcp_valid"] == True)]
    mean_cdd = valid["cdd"].mean()
    slope, p, _ = mk_trend_on_annual(valid, "cdd")
    if slope is not None:
        print(f"  {city:<15} {mean_cdd:>10.1f}  {slope:>+14.2f}  p={p:.4f} {sig_stars(p)}")
    else:
        print(f"  {city:<15} {mean_cdd:>10.1f}  insufficient data")

# ==============================
# 6. DROUGHT YEARS
# ==============================

print_section("6. DROUGHT YEARS  (annual rainfall >= 20% below long-term normal)")
print("  Normal computed from prcp_valid years only.\n")

annual_prcp = (
    master_df.groupby(["city", "year"])["prcp"]
    .sum()
    .reset_index(name="annual_prcp")
    .merge(coverage_df[["city", "year", "prcp_valid"]], on=["city", "year"], how="left")
)

city_normals = (
    annual_prcp[annual_prcp["prcp_valid"] == True]
    .groupby("city")["annual_prcp"]
    .mean()
    .rename("normal_prcp")
)

annual_prcp = annual_prcp.merge(city_normals, on="city")
annual_prcp["pct_of_normal"] = (
    (annual_prcp["annual_prcp"] / annual_prcp["normal_prcp"] - 1) * 100
).round(1)

drought_years = annual_prcp[
    (annual_prcp["prcp_valid"] == True) &
    (annual_prcp["pct_of_normal"] <= DROUGHT_DEFICIT)
].copy()

print(f"  Drought years identified: {len(drought_years)}\n")
print(
    drought_years
    .sort_values(["city", "year"])
    [["city", "year", "annual_prcp", "pct_of_normal"]]
    .rename(columns={"pct_of_normal": "% of normal"})
    .to_string(index=False)
)

print("\n  Drought frequency by city:")
valid_counts   = annual_prcp[annual_prcp["prcp_valid"] == True].groupby("city").size()
drought_counts = drought_years.groupby("city").size().rename("drought_years")
drought_freq   = pd.concat([drought_counts, valid_counts.rename("valid_years")], axis=1).fillna(0)
drought_freq["drought_years"] = drought_freq["drought_years"].astype(int)
drought_freq["frequency_%"]   = (drought_freq["drought_years"] / drought_freq["valid_years"] * 100).round(1)
print(drought_freq.sort_values("frequency_%", ascending=False).to_string())

# ==============================
# SUMMARY
# ==============================

print_section("SUMMARY - KEY EXTREME EVENT FINDINGS")

if not hw_events_df.empty:
    worst_hw = hw_events_df.loc[hw_events_df["duration_days"].idxmax()]
    print(f"\n  Heatwaves:")
    print(f"    Most events   : "
          f"{hw_summary_df.loc[hw_summary_df['total_events'].idxmax(), 'city']}"
          f" ({hw_summary_df['total_events'].max()} events)")
    print(f"    Longest event : {worst_hw['city']} - {worst_hw['duration_days']} days "
          f"({worst_hw['start_date']} to {worst_hw['end_date']})")
    print(f"    Peak tmax     : {hw_events_df['peak_tmax'].max()}C "
          f"({hw_events_df.loc[hw_events_df['peak_tmax'].idxmax(), 'city']})")

top_rain_day = master_df.loc[master_df["prcp"].idxmax()]
print(f"\n  Rainfall:")
print(f"    Wettest single day : {top_rain_day['city']} - "
      f"{top_rain_day['prcp']:.1f}mm on {str(top_rain_day['time'].date())}")
print(f"    Drought years total: {len(drought_years)} across all cities")
print(f"\n  Next step: run 06_visualizations.py")