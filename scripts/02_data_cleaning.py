import pandas as pd
import os

# ==============================
# 1. LOAD DATA
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

# ==============================
# 2. BASIC STRUCTURAL CLEANING
# ==============================

master_df["time"] = pd.to_datetime(master_df["time"], format="mixed", errors="coerce")
master_df = master_df.dropna(subset=["time"])  # Deleting Rows in which the time value is missing

master_df["year"] = master_df["time"].dt.year
master_df["month"] = master_df["time"].dt.month

columns_to_drop = ["snow", "wpgt", "tsun", "pres", "wdir", "wspd"]   # Only available in Rourkela 
master_df = master_df.drop(columns=columns_to_drop)

master_df = master_df.drop_duplicates(subset=["city", "time"])  # To make sure there is no repetition of data

# ==============================
# 3. YEARLY COVERAGE ANALYSIS
# ==============================

coverage_records = []  # To sanity check the data 

for city in master_df["city"].unique():
    city_df = master_df[master_df["city"] == city]
    
    for year in city_df["year"].unique():
        year = int(year)
        year_df = city_df[city_df["year"] == year]
        
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        expected_days = (end - start).days + 1
        
        coverage_records.append({
            "city": city,
            "year": year,
            "expected_days": expected_days,
            "tavg_coverage": year_df["tavg"].notnull().sum() / expected_days * 100,
            "tmin_coverage": year_df["tmin"].notnull().sum() / expected_days * 100,
            "tmax_coverage": year_df["tmax"].notnull().sum() / expected_days * 100,
            "prcp_coverage": year_df["prcp"].notnull().sum() / expected_days * 100
        })

coverage_df = pd.DataFrame(coverage_records)

# ==============================
# 4. MONTHLY RAINFALL COVERAGE
# ==============================

monthly_prcp_coverage = (
    master_df
    .groupby(["city", "year", "month"])["prcp"]
    .apply(lambda x: x.notnull().mean() * 100)
    .reset_index(name="prcp_monthly_coverage")
)

# Extract monsoon months (June–September)
monsoon = monthly_prcp_coverage[
    monthly_prcp_coverage["month"].isin([6, 7, 8, 9])
]

monsoon_min = (
    monsoon
    .groupby(["city", "year"])["prcp_monthly_coverage"]
    .min()
    .reset_index(name="monsoon_min_coverage")
)

coverage_df = coverage_df.merge(
    monsoon_min,
    on=["city", "year"],
    how="left"
)

# ==============================
# 5. VALIDITY FLAGS
# ==============================

coverage_df["temp_valid"] = coverage_df["tavg_coverage"] >= 90  # Marking which data is usable and which is not

coverage_df["prcp_valid"] = (
    (coverage_df["prcp_coverage"] >= 80) &
    (coverage_df["monsoon_min_coverage"] >= 60)
)

# ==============================
# 6. PHYSICAL CONSISTENCY CHECKS
# ==============================

# tmax must be >= tmin
invalid_temp_order = master_df[
    (master_df["tmax"].notnull()) &
    (master_df["tmin"].notnull()) &
    (master_df["tmax"] < master_df["tmin"])
]

negative_rain = master_df[
    (master_df["prcp"].notnull()) &
    (master_df["prcp"] < 0)
]

extreme_temp = master_df[
    (master_df["tavg"] < -15) |
    (master_df["tavg"] > 55) |
    (master_df["tmin"] < -15) |
    (master_df["tmin"] > 55) |
    (master_df["tmax"] < -15) |
    (master_df["tmax"] > 55)
]

#print("Physically implausible temperature rows:", len(extreme_temp))
#print("Negative rainfall rows:", len(negative_rain))
#print("Rows where tmax < tmin:", len(invalid_temp_order))

master_df.loc[                              # Because of that corrupted delhi row which was showing tmin = 0.1 at 6th july
    (master_df["city"] == "Delhi") &
    (master_df["time"] == "2003-07-06"),
    "tmin"
] = pd.NA

master_df["tavg_theoretical"] = (master_df["tmin"] + master_df["tmax"]) / 2

comparison = master_df.dropna(subset=["tavg", "tavg_theoretical"])

comparison["difference"] = comparison["tavg"] - comparison["tavg_theoretical"]

'''print("Mean difference:", comparison["difference"].mean())
print("Std difference:", comparison["difference"].std())
print("Max abs difference:", comparison["difference"].abs().max())'''

# Delhi July climatology       # Because Some values in Delhi at the starting of July felt off physically
delhi_july = master_df[
    (master_df["city"] == "Delhi") &
    (master_df["month"] == 7)
]

mean_july_tmin = delhi_july["tmin"].mean()
std_july_tmin = delhi_july["tmin"].std()


# Compute monthly climatology
monthly_stats = (
    master_df
    .groupby(["city", "month"])["tavg"]
    .agg(["mean", "std"])
    .reset_index()
)

# Merge back
master_df = master_df.merge(
    monthly_stats,
    on=["city", "month"],
    how="left"
)

# Compute z-score
master_df["tavg_z"] = (
    (master_df["tavg"] - master_df["mean"]) /
    master_df["std"]
)

# Flag extreme anomalies
master_df["tavg_extreme_flag"] = master_df["tavg_z"].abs() > 4

print(master_df["tavg_extreme_flag"].sum())

print(master_df[master_df["tavg_extreme_flag"]]
    .groupby("city")
    .size()
    )

print(monthly_stats[monthly_stats["city"] == "Mumbai"])  # Mumbai has the most flagged climatology but it's coastal so it can be possible

