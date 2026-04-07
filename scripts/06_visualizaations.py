import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os
import pymannkendall as mk
from scipy import stats

# ==============================
# SETUP
# ==============================

data_path  = "./data"
fig_dir    = "./figures"
os.makedirs(fig_dir, exist_ok=True)

file_city_map = {
    "Bangalore_1990_2022_BangaloreCity.csv": "Bangalore",
    "Chennai_1990_2022_Madras.csv":          "Chennai",
    "Delhi_NCR_1990_2022_Safdarjung.csv":    "Delhi",
    "Lucknow_1990_2022.csv":                 "Lucknow",
    "Mumbai_1990_2022_Santacruz.csv":        "Mumbai",
    # NOTE: Rajasthan_1990_2022_Jodhpur.csv is a byte-for-byte duplicate of
    # Bangalore_1990_2022_BangaloreCity.csv (confirmed via MD5 hash).
    # Removed to prevent duplicate analysis. Replace with real Jodhpur data when available.
    "weather_Bhubhneshwar_1990_2022.csv":    "Bhubaneswar",
    "weather_Rourkela_2021_2022.csv":        "Rourkela"
}

df_list = []
for file, city in file_city_map.items():
    temp_df = pd.read_csv(os.path.join(data_path, file))
    temp_df["city"] = city
    df_list.append(temp_df)

master_df = pd.concat(df_list, ignore_index=True)
master_df["time"] = pd.to_datetime(master_df["time"], format="mixed", errors="coerce")
master_df = master_df.dropna(subset=["time"])
master_df["year"]  = master_df["time"].dt.year
master_df["month"] = master_df["time"].dt.month
master_df = master_df.drop(columns=["snow", "wpgt", "tsun", "pres", "wdir", "wspd"])
master_df = master_df.drop_duplicates(subset=["city", "time"])
master_df = master_df.sort_values(["city", "time"]).reset_index(drop=True)

master_df.loc[
    (master_df["city"] == "Delhi") & (master_df["time"] == "2003-07-06"), "tmin"
] = pd.NA

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
        year     = int(year)
        year_df  = city_df[city_df["year"] == year]
        start    = pd.Timestamp(f"{year}-01-01")
        end      = pd.Timestamp(f"{year}-12-31")
        exp_days = (end - start).days + 1
        coverage_records.append({
            "city": city, "year": year,
            "tavg_coverage": year_df["tavg"].notnull().sum() / exp_days * 100,
            "prcp_coverage": year_df["prcp"].notnull().sum() / exp_days * 100,
        })

coverage_df = pd.DataFrame(coverage_records)
monthly_prcp_cov = (
    master_df.groupby(["city", "year", "month"])["prcp"]
    .apply(lambda x: x.notnull().mean() * 100)
    .reset_index(name="prcp_monthly_coverage")
)
monsoon_min = (
    monthly_prcp_cov[monthly_prcp_cov["month"].isin([6, 7, 8, 9])]
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

# ==============================
# STYLE
# ==============================

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# Consistent city colours throughout all figures
CITY_COLOR = {
    "Bangalore":   "#1565C0",
    "Bhubaneswar": "#2E7D32",
    "Chennai":     "#E65100",
    "Delhi":       "#6A1B9A",
    "Lucknow":     "#F57F17",
    "Mumbai":      "#00838F",
    # "Rajasthan": removed — duplicate file, no real data
    "Rourkela":    "#4E342E",
}

CITIES_MAIN = ["Bangalore", "Bhubaneswar", "Chennai", "Delhi",
               "Lucknow", "Mumbai"]  # Rourkela excluded (short record); Rajasthan excluded (duplicate file)

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

def save(name):
    path = os.path.join(fig_dir, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {name}")

def add_trend_line(ax, years, values, color="red", lw=1.4, alpha=0.85):
    """Fits OLS and draws a dashed trend line on the given axes."""
    df = pd.DataFrame({"y": years, "v": values}).dropna()
    if len(df) < 5:
        return
    slope, intercept, _, _, _ = stats.linregress(df["y"], df["v"])
    x_line = np.array([df["y"].min(), df["y"].max()])
    ax.plot(x_line, slope * x_line + intercept,
            color=color, lw=lw, linestyle="--", alpha=alpha, zorder=3)

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

print("Building visualizations...\n")

# ======================================================================
# FIGURE 1 — Annual Mean Temperature Time Series (3×3 small multiples)
# One subplot per city. Shows raw annual mean, 5-yr rolling mean,
# and OLS trend line. Annotates Sen's slope + significance.
# ======================================================================

annual_temp = (
    master_df.groupby(["city", "year"])["tavg"]
    .mean()
    .reset_index(name="tavg")
    .merge(coverage_df[["city", "year", "temp_valid"]], on=["city", "year"], how="left")
)

fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=False)
axes = axes.flatten()

for idx, city in enumerate(list(CITY_COLOR.keys())):
    ax  = axes[idx]
    c   = annual_temp[annual_temp["city"] == city].dropna(subset=["tavg"])
    col = CITY_COLOR[city]

    ax.plot(c["year"], c["tavg"], color=col, lw=1.2, alpha=0.5)
    ax.scatter(c["year"], c["tavg"], color=col, s=18, alpha=0.7, zorder=2)

    # 5-year rolling mean
    if len(c) >= 5:
        roll = c.set_index("year")["tavg"].rolling(5, center=True).mean()
        ax.plot(roll.index, roll.values, color=col, lw=2.5, label="5-yr mean")

    # Trend line (valid years only)
    valid = c[c["temp_valid"] == True]
    if len(valid) >= 10:
        mk_r = mk.original_test(valid["tavg"].values)
        add_trend_line(ax, valid["year"].values, valid["tavg"].values,
                       color="black", lw=1.6)
        slope_label = (f"Sen: {mk_r.slope*10:+.3f}°C/dec "
                       f"{sig_label(mk_r.p)}")
        ax.set_title(f"{city}\n{slope_label}", fontsize=9,
                     color=col, fontweight="bold")
    else:
        ax.set_title(city, fontsize=9, color=col, fontweight="bold")

    ax.set_ylabel("Temp (°C)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

# Hide unused subplot (3×3 grid, 8 cities)
axes[-1].set_visible(False)

fig.suptitle("Annual Mean Temperature 1990–2022\n(dashed = OLS trend, thick = 5-yr rolling mean)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
save("fig01_annual_temperature_timeseries.png")

# ======================================================================
# FIGURE 2 — Warming Rate Comparison (horizontal bar chart)
# Sen's slope per city. Bars coloured by city. Stars for significance.
# Split into temperature (top) and tmin/tmax asymmetry (bottom).
# ======================================================================

temp_trends, tmin_trends, tmax_trends = [], [], []

for city in CITIES_MAIN:
    valid = annual_temp[(annual_temp["city"] == city) &
                        (annual_temp["temp_valid"] == True)].dropna(subset=["tavg"])
    if len(valid) >= 10:
        r = mk.original_test(valid["tavg"].values)
        temp_trends.append({"city": city, "slope": r.slope*10, "p": r.p})

    for var, store in [("tmin", tmin_trends), ("tmax", tmax_trends)]:
        vd = (master_df.groupby(["city", "year"])[var].mean().reset_index(name=var)
              .merge(coverage_df[["city", "year", "temp_valid"]], on=["city", "year"], how="left"))
        sub = vd[(vd["city"] == city) & (vd["temp_valid"] == True)].dropna(subset=[var])
        if len(sub) >= 10:
            r = mk.original_test(sub[var].values)
            store.append({"city": city, "slope": r.slope*10, "p": r.p})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: annual mean warming rates
tt_df = pd.DataFrame(temp_trends).sort_values("slope", ascending=True)
colors = [CITY_COLOR[c] for c in tt_df["city"]]
bars   = ax1.barh(tt_df["city"], tt_df["slope"], color=colors, alpha=0.85, height=0.6)
for bar, (_, row) in zip(bars, tt_df.iterrows()):
    x   = row["slope"]
    lbl = f"{x:+.3f} {sig_label(row['p'])}"
    ax1.text(x + 0.005, bar.get_y() + bar.get_height()/2,
             lbl, va="center", ha="left", fontsize=9)
ax1.axvline(0, color="black", lw=0.8)
ax1.set_xlabel("Sen's Slope (°C per decade)")
ax1.set_title("Annual Mean Temperature\nWarming Rate", fontweight="bold")
ax1.grid(axis="x", alpha=0.25, lw=0.5)

# Right: Tmin vs Tmax asymmetry (grouped bars)
tmin_df = pd.DataFrame(tmin_trends).set_index("city")
tmax_df = pd.DataFrame(tmax_trends).set_index("city")
cities_both = [c for c in CITIES_MAIN if c in tmin_df.index and c in tmax_df.index]

y       = np.arange(len(cities_both))
h       = 0.35
tmin_v  = [tmin_df.loc[c, "slope"] for c in cities_both]
tmax_v  = [tmax_df.loc[c, "slope"] for c in cities_both]

ax2.barh(y - h/2, tmin_v, height=h, color="#1565C0", alpha=0.8, label="Tmin (night)")
ax2.barh(y + h/2, tmax_v, height=h, color="#E65100", alpha=0.8, label="Tmax (day)")
ax2.set_yticks(y)
ax2.set_yticklabels(cities_both)
ax2.axvline(0, color="black", lw=0.8)
ax2.set_xlabel("Sen's Slope (°C per decade)")
ax2.set_title("Tmin vs Tmax Asymmetry\n(nighttime vs daytime warming)", fontweight="bold")
ax2.legend()
ax2.grid(axis="x", alpha=0.25, lw=0.5)

# Highlight Lucknow asymmetry — the standout finding
if "Lucknow" in cities_both:
    idx = cities_both.index("Lucknow")
    ax2.annotate("Lucknow: tmin rising,\ntmax falling",
                 xy=(tmin_df.loc["Lucknow", "slope"], idx - h/2),
                 xytext=(0.3, idx + 1.2),
                 fontsize=8, color="#F57F17",
                 arrowprops=dict(arrowstyle="->", color="#F57F17", lw=1))

plt.tight_layout()
save("fig02_warming_rates_and_asymmetry.png")

# ======================================================================
# FIGURE 3 — Seasonal Temperature Heatmap
# Rows = cities, Columns = seasons. Cell = Sen's slope.
# Colour scale: white → deep red. Annotated with value + stars.
# ======================================================================

seasons_order = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
seasonal_temp = (
    master_df.groupby(["city", "year", "season"])["tavg"]
    .mean()
    .reset_index(name="tavg")
    .merge(coverage_df[["city", "year", "temp_valid"]], on=["city", "year"], how="left")
)

grid_vals = np.full((len(CITIES_MAIN), 4), np.nan)
grid_sigs = [[""] * 4 for _ in CITIES_MAIN]

for i, city in enumerate(CITIES_MAIN):
    for j, season in enumerate(seasons_order):
        sub = seasonal_temp[
            (seasonal_temp["city"] == city) &
            (seasonal_temp["season"] == season) &
            (seasonal_temp["temp_valid"] == True)
        ].dropna(subset=["tavg"])
        if len(sub) >= 10:
            r = mk.original_test(sub["tavg"].values)
            grid_vals[i, j] = r.slope * 10
            grid_sigs[i][j] = sig_label(r.p)

fig, ax = plt.subplots(figsize=(9, 6))
vmax = np.nanmax(np.abs(grid_vals))
im   = ax.imshow(grid_vals, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")

ax.set_xticks(range(4))
ax.set_xticklabels(seasons_order, fontsize=10)
ax.set_yticks(range(len(CITIES_MAIN)))
ax.set_yticklabels(CITIES_MAIN, fontsize=10)

for i in range(len(CITIES_MAIN)):
    for j in range(4):
        v   = grid_vals[i, j]
        sig = grid_sigs[i][j]
        if not np.isnan(v):
            txt_col = "white" if v > vmax * 0.65 else "black"
            ax.text(j, i, f"{v:+.2f}\n{sig}",
                    ha="center", va="center", fontsize=8.5, color=txt_col)
        else:
            ax.text(j, i, "—", ha="center", va="center",
                    fontsize=10, color="gray")

plt.colorbar(im, ax=ax, label="Sen's Slope (°C/decade)", shrink=0.8)
ax.set_title("Seasonal Warming Rate by City (°C per decade)\n"
             "*** p<0.001  ** p<0.01  * p<0.05  ns not significant",
             fontweight="bold", pad=14)
plt.tight_layout()
save("fig03_seasonal_warming_heatmap.png")

# ======================================================================
# FIGURE 4 — Monthly Temperature Anomaly Heatmap  (Year × Month)
# One subplot per city. Red = warmer than normal, Blue = cooler.
# Shows WHEN the warming signal emerged, not just how fast.
# ======================================================================

monthly_clim = (
    master_df.groupby(["city", "month"])["tavg"]
    .mean().rename("clim").reset_index()
)
master_df = master_df.merge(monthly_clim, on=["city", "month"], how="left")
master_df["anomaly"] = master_df["tavg"] - master_df["clim"]

fig, axes = plt.subplots(4, 2, figsize=(15, 18))
axes = axes.flatten()

for idx, city in enumerate(list(CITY_COLOR.keys())):
    ax   = axes[idx]
    c    = master_df[master_df["city"] == city]
    piv  = c.pivot_table(index="year", columns="month",
                         values="anomaly", aggfunc="mean")
    piv.columns = MONTH_LABELS[:len(piv.columns)]
    vmax = max(abs(piv.values[~np.isnan(piv.values)]).max(), 0.5) if piv.size else 1

    im = ax.imshow(piv.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index.astype(int), fontsize=6)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=8)
    ax.set_title(city, fontweight="bold", color=CITY_COLOR[city], fontsize=10)
    plt.colorbar(im, ax=ax, label="Anomaly (°C)", shrink=0.75, pad=0.02)

fig.suptitle("Monthly Temperature Anomaly — Departure from City Climatology\n"
             "(Red = warmer than normal, Blue = cooler)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
save("fig04_monthly_anomaly_heatmap.png")

# ======================================================================
# FIGURE 5 — Seasonal Cycle (monthly mean temperature + rainfall)
# Left: temperature. Right: rainfall.
# Each city is one line, labelled at the end.
# ======================================================================

monthly_temp = master_df.groupby(["city", "month"])["tavg"].mean().reset_index()
monthly_prcp = master_df.groupby(["city", "month"])["prcp"].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for city in CITIES_MAIN:
    col = CITY_COLOR[city]
    t   = monthly_temp[monthly_temp["city"] == city]
    p   = monthly_prcp[monthly_prcp["city"] == city]

    ax1.plot(t["month"], t["tavg"], color=col, lw=2, marker="o",
             markersize=4, label=city)
    ax2.plot(p["month"], p["prcp"], color=col, lw=2, marker="o",
             markersize=4, label=city)

for ax, title, ylabel in [
    (ax1, "Mean Monthly Temperature", "Temperature (°C)"),
    (ax2, "Mean Daily Rainfall", "Rainfall (mm/day)")
]:
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right" if ax == ax1 else "upper left")
    ax.grid(alpha=0.25, lw=0.5)

    # Shade monsoon months
    ax.axvspan(5.5, 9.5, alpha=0.07, color="blue", label="Monsoon (Jun–Sep)")

fig.suptitle("Seasonal Cycle by City (1990–2022 climatology)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
save("fig05_seasonal_cycle.png")

# ======================================================================
# FIGURE 6 — Annual Rainfall Time Series
# Bar chart per city with 5-yr rolling mean overlay.
# Long-term normal shown as a grey dashed line.
# ======================================================================

annual_prcp = (
    master_df.groupby(["city", "year"])["prcp"]
    .sum()
    .reset_index(name="prcp")
    .merge(coverage_df[["city", "year", "prcp_valid"]], on=["city", "year"], how="left")
)

fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=False)
axes = axes.flatten()

for idx, city in enumerate(list(CITY_COLOR.keys())):
    ax  = axes[idx]
    c   = annual_prcp[annual_prcp["city"] == city].dropna(subset=["prcp"])
    col = CITY_COLOR[city]

    # Valid years solid, invalid years hatched
    valid   = c[c["prcp_valid"] == True]
    invalid = c[c["prcp_valid"] != True]

    ax.bar(invalid["year"], invalid["prcp"], color=col, alpha=0.25,
           width=0.8, label="Low coverage year")
    ax.bar(valid["year"],   valid["prcp"],   color=col, alpha=0.75,
           width=0.8, label="Valid year")

    # Long-term normal
    if len(valid) > 0:
        normal = valid["prcp"].mean()
        ax.axhline(normal, color="black", lw=1.2, linestyle="--",
                   alpha=0.6, label=f"Normal: {normal:.0f}mm")

    # 5-yr rolling mean
    if len(c) >= 5:
        roll = c.set_index("year")["prcp"].rolling(5, center=True).mean()
        ax.plot(roll.index, roll.values, color="black", lw=2, alpha=0.8)

    ax.set_title(city, fontweight="bold", color=col, fontsize=9)
    ax.set_ylabel("Rainfall (mm)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

axes[-1].set_visible(False)
fig.suptitle("Annual Rainfall 1990–2022\n"
             "(dark bars = valid years, faded = low coverage, dashed = long-term normal)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
save("fig06_annual_rainfall_timeseries.png")

# ======================================================================
# FIGURE 7 — Heatwave Event Timeline
# Each heatwave = a horizontal bar on a city × time axis.
# Bar length = duration, colour intensity = peak tmax.
# Immediately shows that Delhi and Lucknow dominate.
# ======================================================================

HEATWAVE_TMAX     = 40.0
HEATWAVE_MIN_DAYS = 3

def extract_heatwave_events(city_df):
    events = []
    rows   = city_df.dropna(subset=["tmax"]).reset_index(drop=True)
    i = 0
    while i < len(rows):
        if rows.loc[i, "tmax"] >= HEATWAVE_TMAX:
            j = i
            while j < len(rows) and rows.loc[j, "tmax"] >= HEATWAVE_TMAX:
                if j > i and (rows.loc[j,"time"] - rows.loc[j-1,"time"]).days > 1:
                    break
                j += 1
            dur = j - i
            if dur >= HEATWAVE_MIN_DAYS:
                events.append({
                    "start": rows.loc[i, "time"],
                    "end":   rows.loc[j-1, "time"],
                    "dur":   dur,
                    "peak":  rows.loc[i:j-1, "tmax"].max(),
                    "year":  rows.loc[i, "year"],
                })
            i = j
        else:
            i += 1
    return events

hw_cities = ["Delhi", "Lucknow", "Chennai", "Bhubaneswar"]
city_events = {}
for city in hw_cities:
    city_df = master_df[master_df["city"] == city].reset_index(drop=True)
    city_events[city] = extract_heatwave_events(city_df)

fig, ax = plt.subplots(figsize=(14, 5))

import matplotlib.cm as cm
cmap   = cm.YlOrRd
norm   = plt.Normalize(vmin=40, vmax=48)
y_pos  = {city: i for i, city in enumerate(hw_cities)}

for city, events in city_events.items():
    y = y_pos[city]
    for e in events:
        color = cmap(norm(e["peak"]))
        ax.barh(y, (e["end"] - e["start"]).days + 1,
                left=e["start"], height=0.6,
                color=color, alpha=0.85, edgecolor="none")

ax.set_yticks(range(len(hw_cities)))
ax.set_yticklabels(hw_cities, fontsize=11)
ax.set_xlabel("Year")
ax.set_title("Heatwave Event Timeline (tmax ≥ 40°C for ≥ 3 consecutive days)\n"
             "Colour = peak tmax (yellow → dark red)", fontweight="bold")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Peak tmax (°C)", shrink=0.7, pad=0.02)
ax.grid(axis="x", alpha=0.2, lw=0.5)
plt.tight_layout()
save("fig07_heatwave_timeline.png")

# ======================================================================
# FIGURE 8 — Tmin/Tmax Divergence: Lucknow Deep Dive
# The standout finding from script 04.
# Single figure showing both tmin and tmax annual means for Lucknow
# with individual trend lines, highlighting the divergence after ~2005.
# ======================================================================

lucknow = (
    master_df[master_df["city"] == "Lucknow"]
    .groupby("year")[["tmin", "tmax", "tavg"]]
    .mean()
    .reset_index()
    .merge(coverage_df[coverage_df["city"] == "Lucknow"][["year", "temp_valid"]],
           on="year", how="left")
)

fig, ax = plt.subplots(figsize=(11, 5))

col_tmin = "#1565C0"
col_tmax = "#C62828"
col_tavg = "#F57F17"

ax.plot(lucknow["year"], lucknow["tmin"], color=col_tmin, lw=1.5,
        alpha=0.6, marker="o", markersize=4)
ax.plot(lucknow["year"], lucknow["tmax"], color=col_tmax, lw=1.5,
        alpha=0.6, marker="o", markersize=4)
ax.plot(lucknow["year"], lucknow["tavg"], color=col_tavg, lw=1.5,
        alpha=0.6, marker="o", markersize=4)

# Trend lines on valid years only
for col, var, label in [
    (col_tmin, "tmin", "Tmin"),
    (col_tmax, "tmax", "Tmax"),
    (col_tavg, "tavg", "Tavg"),
]:
    valid = lucknow[lucknow["temp_valid"] == True].dropna(subset=[var])
    if len(valid) >= 10:
        r = mk.original_test(valid[var].values)
        add_trend_line(ax, valid["year"].values, valid[var].values,
                       color=col, lw=2.2)
        slope_txt = f"{label}: {r.slope*10:+.3f}°C/dec ({sig_label(r.p)})"
        ax.plot([], [], color=col, lw=2, linestyle="--", label=slope_txt)

ax.set_xlabel("Year")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Lucknow — Tmin vs Tmax Divergence 1990–2022\n"
             "Nighttime warming (+0.46°C/dec) vs daytime cooling (−0.31°C/dec)",
             fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, lw=0.5)

# Annotate the divergence zone
ax.axvspan(2005, 2022, alpha=0.06, color="#F57F17")
ax.text(2005.5, lucknow["tmax"].max() - 0.3,
        "Divergence\naccelerates",
        fontsize=8.5, color="#F57F17", va="top")

plt.tight_layout()
save("fig08_lucknow_tmin_tmax_divergence.png")

# ======================================================================
# FIGURE 9 — Warm Nights Trend (TN90p)
# Annual warm night count per city with trend line.
# Connects the nighttime warming finding to a health-relevant metric.
# ======================================================================

tmin_pctile = (
    master_df.groupby(["city", "month"])["tmin"]
    .quantile(0.90).rename("tmin_p90").reset_index()
)
master_df = master_df.merge(tmin_pctile, on=["city", "month"], how="left")
master_df["warm_night"] = master_df["tmin"] > master_df["tmin_p90"]

annual_wn = (
    master_df.groupby(["city", "year"])["warm_night"]
    .sum().reset_index(name="warm_nights")
    .merge(coverage_df[["city", "year", "temp_valid"]], on=["city", "year"], how="left")
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=False)
axes = axes.flatten()

for idx, city in enumerate(list(CITY_COLOR.keys())):
    ax  = axes[idx]
    c   = annual_wn[annual_wn["city"] == city]
    col = CITY_COLOR[city]

    ax.bar(c["year"], c["warm_nights"], color=col, alpha=0.6, width=0.8)

    valid = c[c["temp_valid"] == True].dropna(subset=["warm_nights"])
    if len(valid) >= 10:
        r = mk.original_test(valid["warm_nights"].values)
        add_trend_line(ax, valid["year"].values,
                       valid["warm_nights"].values.astype(float),
                       color="black", lw=1.8)
        ax.set_title(f"{city}\n{r.slope*10:+.1f} days/dec {sig_label(r.p)}",
                     fontsize=9, color=col, fontweight="bold")
    else:
        ax.set_title(f"{city}\n(insufficient data)", fontsize=9,
                     color=col, fontweight="bold")

    ax.set_ylabel("Warm nights/yr", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

fig.suptitle("Annual Warm Nights (TN90p — tmin > 90th percentile)\n"
             "Trend = dashed black line",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
save("fig09_warm_nights_trend.png")

# ======================================================================
# FIGURE 10 — Decade Warming Summary (dot plot)
# Each city gets a dot for 1990s mean and 2010s mean,
# connected by a line. Length of line = warming magnitude.
# Clean, immediately readable for a paper.
# ======================================================================

master_df["decade"] = (master_df["year"] // 10 * 10).astype(str) + "s"
decade_temp = (
    master_df.groupby(["city", "decade"])["tavg"]
    .mean().reset_index()
)

fig, ax = plt.subplots(figsize=(10, 6))

cities_plot = [c for c in CITIES_MAIN
               if "1990s" in decade_temp[decade_temp["city"] == c]["decade"].values
               and "2010s" in decade_temp[decade_temp["city"] == c]["decade"].values]

for i, city in enumerate(cities_plot):
    col   = CITY_COLOR[city]
    dec   = decade_temp[decade_temp["city"] == city].set_index("decade")["tavg"]
    t90   = dec.get("1990s", np.nan)
    t00   = dec.get("2000s", np.nan)
    t10   = dec.get("2010s", np.nan)

    if not np.isnan(t90) and not np.isnan(t10):
        ax.plot([t90, t10], [i, i], color=col, lw=2.5, alpha=0.6, zorder=1)
        ax.scatter([t90], [i], color=col, s=80,  marker="o",
                   zorder=2, label="_nolegend_")
        ax.scatter([t10], [i], color=col, s=120, marker="D",
                   zorder=2, label="_nolegend_")
        delta = t10 - t90
        ax.text(max(t90, t10) + 0.04, i, f"+{delta:.2f}°C",
                va="center", ha="left", fontsize=9, color=col, fontweight="bold")

ax.set_yticks(range(len(cities_plot)))
ax.set_yticklabels(cities_plot, fontsize=11)
ax.set_xlabel("Mean Annual Temperature (°C)")
ax.set_title("Decade Shift: 1990s → 2010s\n"
             "Circle = 1990s mean, Diamond = 2010s mean",
             fontweight="bold")
ax.grid(axis="x", alpha=0.25, lw=0.5)

# Legend
ax.scatter([], [], color="gray", s=80,  marker="o",  label="1990s mean")
ax.scatter([], [], color="gray", s=120, marker="D",  label="2010s mean")
ax.legend(fontsize=9)
plt.tight_layout()
save("fig10_decade_warming_dotplot.png")

# ======================================================================
# DONE
# ======================================================================

print(f"\nAll 10 figures saved to ./{fig_dir}/")
print("""
  fig01 — Annual temperature time series (small multiples)
  fig02 — Warming rates + tmin/tmax asymmetry
  fig03 — Seasonal warming heatmap
  fig04 — Monthly anomaly heatmap (year × month)
  fig05 — Seasonal cycle (temp + rainfall)
  fig06 — Annual rainfall time series
  fig07 — Heatwave event timeline
  fig08 — Lucknow tmin/tmax divergence (deep dive)
  fig09 — Warm nights (TN90p) trend
  fig10 — Decade shift dot plot
""")