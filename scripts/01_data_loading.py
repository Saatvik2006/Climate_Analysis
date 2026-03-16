import pandas as pd
import os

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
master_df["year"] = master_df["time"].dt.year
master_df["month"] = master_df["time"].dt.month

columns_to_drop = ["snow", "wpgt", "tsun", "pres", "wdir", "wspd"]

master_df = master_df.drop(columns=columns_to_drop)

print(master_df.columns)

print(master_df["year"].min(), master_df["year"].max())
print(master_df.isnull().sum())
missing_percent = master_df.isnull().mean() * 100
print(missing_percent.sort_values(ascending=False))
coverage = master_df.groupby("city").agg({
    "tavg": lambda x: x.notnull().mean() * 100,
    "tmin": lambda x: x.notnull().mean() * 100,
    "tmax": lambda x: x.notnull().mean() * 100,
    "prcp": lambda x: x.notnull().mean() * 100
})

print(coverage)