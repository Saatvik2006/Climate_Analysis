import pandas as pd
import numpy as np
import os
from scipy import stats
import pymannkendall as mk

# ==============================
# SETUP — reuse cleaned data
# ==============================

data_path = "./data"

file_city_map = {
    "Bangalore_1990_2022_BangaloreCity.csv": "Bangalore",
    "Chennai_1990_2022_Madras.csv": "Chennai",
    "Delhi_NCR_1990_2022_Safdarjung.csv": "Delhi",
    "Lucknow_1990_2022.csv": "Lucknow",
    "Mumbai_1990_2022_Santacruz.csv": "Mumbai",
    "Rajasthan_1990_2022_Jodhpur.csv": "Rajasthan",
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

# Corrupted Delhi row fix from script 02
master_df.loc[
    (master_df["city"] == "Delhi") & (master_df["time"] == "2003-07-06"), "tmin"
] = pd.NA

# Season labels
season_map = {
    12: "Winter",      1: "Winter",      2: "Winter",
    3:  "Pre-Monsoon", 4: "Pre-Monsoon", 5: "Pre-Monsoon",
    6:  "Monsoon",     7: "Monsoon",     8: "Monsoon",     9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon"
}
master_df["season"] = master_df["month"].map(season_map)

# ==============================
# COVERAGE FLAGS (from script 02)
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

MIN_YEARS = 10  # minimum valid years to run any trend test

# ==============================
# CORE TREND FUNCTION
# ==============================

def run_trend_analysis(series, years, label=""):
    """
    Runs three complementary trend methods on an annual time series.

    Methods
    -------
    1. Mann-Kendall (original_test from pymannkendall)
       - Non-parametric: no normality assumption
       - Detects monotonic trends, robust to outliers
       - Also returns Sen's Slope (median pairwise slope)
       - p < 0.05 = statistically significant trend

    2. Sen's Slope (returned by pymannkendall alongside MK)
       - Median of all pairwise slopes between data points
       - Far more robust than OLS when outliers are present
       - Primary magnitude estimate reported in climate papers

    3. OLS linear regression (via scipy.stats.linregress)
       - Ordinary Least Squares — standard regression
       - Sensitive to outliers, but gives R² and is universally comparable
       - Use OLS R² to show how much variance the trend explains
       - Use OLS p-value as a cross-check alongside MK

    Parameters
    ----------
    series : pd.Series  — annual values (e.g. mean tavg per year)
    years  : pd.Series  — corresponding years
    label  : str        — optional, for display only

    Returns None if fewer than MIN_YEARS valid data points.
    """
    df = pd.DataFrame({"year": years, "value": series}).dropna().sort_values("year")

    if len(df) < MIN_YEARS:
        return None

    x = df["year"].values.astype(float)
    y = df["value"].values.astype(float)

    # Mann-Kendall + Sen's Slope (pymannkendall handles both together)
    mk_result = mk.original_test(y)

    # OLS
    slope_ols, _, r, p_ols, _ = stats.linregress(x, y)

    return {
        "n_years"              : len(df),
        "mk_trend"             : mk_result.trend,          # "increasing" / "decreasing" / "no trend"
        "mk_tau"               : round(mk_result.Tau, 4),  # strength: -1 to +1
        "mk_p"                 : round(mk_result.p, 4),    # significance
        "mk_significant"       : mk_result.p < 0.05,
        "sens_slope_per_decade": round(mk_result.slope * 10, 4),  # primary magnitude
        "ols_slope_per_decade" : round(slope_ols * 10, 4),        # cross-check
        "ols_p"                : round(p_ols, 4),
        "ols_r2"               : round(r ** 2, 4),                # variance explained
    }


def sig_stars(p):
    """Standard significance stars used in climate papers."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def print_section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ==============================
# 1. ANNUAL MEAN TEMPERATURE TRENDS
# ==============================

print_section("1. ANNUAL MEAN TEMPERATURE TRENDS")
print("  Valid years only (tavg coverage >= 90%)")
print("  Sen's slope = primary magnitude | MK = significance | OLS R² = fit quality")
print("  Stars: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant\n")

annual_temp = (
    master_df.groupby(["city", "year"])["tavg"]
    .mean()
    .reset_index(name="tavg")
    .merge(coverage_df[["city", "year", "temp_valid"]], on=["city", "year"], how="left")
)

temp_results = []
for city in sorted(annual_temp["city"].unique()):
    valid = annual_temp[
        (annual_temp["city"] == city) & (annual_temp["temp_valid"] == True)
    ]
    result = run_trend_analysis(valid["tavg"], valid["year"], city)

    if result is None:
        print(f"  {city:<15} — skipped (< {MIN_YEARS} valid years)")
        continue

    print(
        f"  {city:<15} "
        f"Sen: {result['sens_slope_per_decade']:+.3f} °C/dec  "
        f"MK p={result['mk_p']:.4f} {sig_stars(result['mk_p']):>3}  "
        f"tau={result['mk_tau']:+.3f}  "
        f"OLS R²={result['ols_r2']:.3f}  "
        f"n={result['n_years']}"
    )
    temp_results.append({"city": city, **result})

temp_results_df = pd.DataFrame(temp_results)

print("\n  Warming rate ranking:")
for _, row in temp_results_df.sort_values("sens_slope_per_decade", ascending=False).iterrows():
    bar = "█" * max(1, int(abs(row["sens_slope_per_decade"]) * 25))
    print(f"    {row['city']:<15} {row['sens_slope_per_decade']:+.3f} °C/dec  {bar}  {sig_stars(row['mk_p'])}")

# ==============================
# 2. TEMPERATURE ANOMALY TRENDS
# Subtracts monthly climatology from each observation.
# Removes the seasonal cycle so you can compare the warming signal
# directly across cities with very different base temperatures
# (e.g. Rajasthan at ~28°C vs Bangalore at ~23°C baseline).
# ==============================

print_section("2. TEMPERATURE ANOMALY TRENDS")
print("  Anomaly = daily tavg minus city's long-term monthly mean.")
print("  Removes seasonal cycle. Lets you compare warming signal across cities.\n")

monthly_clim = (
    master_df.groupby(["city", "month"])["tavg"]
    .mean()
    .rename("clim_tavg")
    .reset_index()
)
master_df = master_df.merge(monthly_clim, on=["city", "month"], how="left")
master_df["tavg_anomaly"] = master_df["tavg"] - master_df["clim_tavg"]

annual_anomaly = (
    master_df.groupby(["city", "year"])["tavg_anomaly"]
    .mean()
    .reset_index(name="tavg_anomaly")
    .merge(coverage_df[["city", "year", "temp_valid"]], on=["city", "year"], how="left")
)

for city in sorted(annual_anomaly["city"].unique()):
    valid = annual_anomaly[
        (annual_anomaly["city"] == city) & (annual_anomaly["temp_valid"] == True)
    ]
    result = run_trend_analysis(valid["tavg_anomaly"], valid["year"], city)
    if result is None:
        print(f"  {city:<15} — skipped")
        continue
    print(
        f"  {city:<15} "
        f"Anomaly trend: {result['sens_slope_per_decade']:+.3f} °C/dec  "
        f"MK p={result['mk_p']:.4f} {sig_stars(result['mk_p']):>3}  "
        f"OLS R²={result['ols_r2']:.3f}"
    )

# ==============================
# 3. SEASONAL TEMPERATURE TRENDS
# Warming is not uniform across seasons.
# Pre-monsoon warming = heatwave risk.
# Winter warming = crop cycle and cold stress changes.
# Monsoon warming = evaporation and rainfall intensity changes.
# ==============================

print_section("3. SEASONAL TEMPERATURE TRENDS  (Sen's slope °C/decade)")
print("  Which season is warming fastest in each city?\n")

seasons_ordered = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]

seasonal_temp = (
    master_df.groupby(["city", "year", "season"])["tavg"]
    .mean()
    .reset_index(name="tavg")
    .merge(coverage_df[["city", "year", "temp_valid"]], on=["city", "year"], how="left")
)

seasonal_records = []
for city in sorted(seasonal_temp["city"].unique()):
    row = {"city": city}
    for season in seasons_ordered:
        subset = seasonal_temp[
            (seasonal_temp["city"] == city) &
            (seasonal_temp["season"] == season) &
            (seasonal_temp["temp_valid"] == True)
        ]
        result = run_trend_analysis(subset["tavg"], subset["year"])
        if result:
            row[season] = f"{result['sens_slope_per_decade']:+.3f}{sig_stars(result['mk_p'])}"
        else:
            row[season] = "—"
    seasonal_records.append(row)

seasonal_df = pd.DataFrame(seasonal_records).set_index("city")
print(seasonal_df.to_string())
print("\n  Format: slope °C/decade + significance  |  — = insufficient data")

# ==============================
# 4. TMIN vs TMAX ASYMMETRY
# If tmin (nighttime low) warms faster than tmax (daytime high):
#   → diurnal temperature range is narrowing
#   → typical signature of urban heat island + greenhouse effect
#   → important for health impact studies (no nighttime relief from heat)
# ==============================

print_section("4. TMIN vs TMAX ASYMMETRY")
print("  Nighttime warming (tmin) faster than daytime (tmax)")
print("  = narrowing diurnal range = urban heat island signal\n")

for var, label in [("tmin", "Tmin (nighttime low)"), ("tmax", "Tmax (daytime high)")]:
    annual_var = (
        master_df.groupby(["city", "year"])[var]
        .mean()
        .reset_index(name=var)
        .merge(coverage_df[["city", "year", "temp_valid"]], on=["city", "year"], how="left")
    )
    print(f"  {label}:")
    for city in sorted(annual_var["city"].unique()):
        valid = annual_var[
            (annual_var["city"] == city) & (annual_var["temp_valid"] == True)
        ]
        result = run_trend_analysis(valid[var], valid["year"])
        if result:
            print(f"    {city:<15} {result['sens_slope_per_decade']:+.3f} °C/dec  {sig_stars(result['mk_p'])}")
    print()

# ==============================
# 5. ANNUAL RAINFALL TRENDS
# ==============================

print_section("5. ANNUAL RAINFALL TRENDS")
print("  Valid years only (prcp coverage >= 80% AND monsoon months >= 60%)\n")

annual_prcp = (
    master_df.groupby(["city", "year"])["prcp"]
    .sum()
    .reset_index(name="prcp")
    .merge(coverage_df[["city", "year", "prcp_valid"]], on=["city", "year"], how="left")
)

for city in sorted(annual_prcp["city"].unique()):
    valid = annual_prcp[
        (annual_prcp["city"] == city) & (annual_prcp["prcp_valid"] == True)
    ]
    result = run_trend_analysis(valid["prcp"], valid["year"], city)
    if result is None:
        print(f"  {city:<15} — skipped (< {MIN_YEARS} valid years)")
        continue
    print(
        f"  {city:<15} "
        f"Sen: {result['sens_slope_per_decade']:+.1f} mm/dec  "
        f"MK p={result['mk_p']:.4f} {sig_stars(result['mk_p']):>3}  "
        f"OLS R²={result['ols_r2']:.3f}  "
        f"n={result['n_years']}"
    )

# ==============================
# 6. MONSOON RAINFALL TRENDS
# Even if annual totals are stable, a declining monsoon with
# compensating pre/post-monsoon rains signals redistribution —
# a critical risk for agriculture and water security.
# ==============================

print_section("6. MONSOON RAINFALL TRENDS  (June–September only)")
print("  A declining monsoon trend even with stable annual total = redistribution risk.\n")

monsoon_prcp = (
    master_df[master_df["season"] == "Monsoon"]
    .groupby(["city", "year"])["prcp"]
    .sum()
    .reset_index(name="prcp")
    .merge(coverage_df[["city", "year", "prcp_valid"]], on=["city", "year"], how="left")
)

for city in sorted(monsoon_prcp["city"].unique()):
    valid = monsoon_prcp[
        (monsoon_prcp["city"] == city) & (monsoon_prcp["prcp_valid"] == True)
    ]
    result = run_trend_analysis(valid["prcp"], valid["year"], city)
    if result is None:
        print(f"  {city:<15} — skipped")
        continue
    print(
        f"  {city:<15} "
        f"Sen: {result['sens_slope_per_decade']:+.1f} mm/dec  "
        f"MK p={result['mk_p']:.4f} {sig_stars(result['mk_p']):>3}  "
        f"OLS R²={result['ols_r2']:.3f}"
    )

# ==============================
# 7. DECADE-OVER-DECADE SHIFT
# No statistics — just raw decadal averages.
# Powerful for a paper because it's immediately interpretable:
# "Delhi was 0.6°C warmer in the 2010s than the 1990s."
# ==============================

print_section("7. DECADE-OVER-DECADE TEMPERATURE SHIFT  (mean annual tavg, °C)")
print("  Raw decadal averages — no model needed. Directly quotable in a paper.\n")

master_df["decade"] = (master_df["year"] // 10 * 10).astype(str) + "s"

decade_temp = (
    master_df.groupby(["city", "decade"])["tavg"]
    .mean()
    .round(2)
    .unstack("decade")
)
if "1990s" in decade_temp.columns and "2010s" in decade_temp.columns:
    decade_temp["Δ 2010s−1990s"] = (decade_temp["2010s"] - decade_temp["1990s"]).round(2)

print(decade_temp.to_string())

print_section("7b. DECADE-OVER-DECADE RAINFALL SHIFT  (mean annual total, mm)")

decade_prcp = (
    master_df.groupby(["city", "decade", "year"])["prcp"]
    .sum()
    .groupby(["city", "decade"])
    .mean()
    .round(1)
    .unstack("decade")
)
if "1990s" in decade_prcp.columns and "2010s" in decade_prcp.columns:
    decade_prcp["Δ 2010s−1990s"] = (decade_prcp["2010s"] - decade_prcp["1990s"]).round(1)

print(decade_prcp.to_string())

# ==============================
# SUMMARY
# ==============================

print_section("SUMMARY — KEY FINDINGS")
if not temp_results_df.empty:
    fastest = temp_results_df.loc[temp_results_df["sens_slope_per_decade"].idxmax()]
    sig_cities = temp_results_df[temp_results_df["mk_significant"]]["city"].tolist()
    insig_cities = temp_results_df[~temp_results_df["mk_significant"]]["city"].tolist()
    print(f"  Fastest warming : {fastest['city']} "
          f"({fastest['sens_slope_per_decade']:+.3f} °C/decade, "
          f"p={fastest['mk_p']:.4f})")
    print(f"  Significant (p<0.05) : {', '.join(sig_cities) if sig_cities else 'none'}")
    print(f"  Not significant      : {', '.join(insig_cities) if insig_cities else 'none'}")
    print(f"  Cities analysed      : {len(temp_results_df)}")
    print(f"\n  Next step → run 05_extreme_events.py")