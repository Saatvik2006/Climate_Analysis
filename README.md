#  Climate Analysis Project (1990–2022)

This project presents a data-driven analysis of long-term climate trends across major Indian cities using statistical and visualization techniques.

---

##  Overview

The objective of this project is to demonstrate how data analysis techniques can be used to:

- Identify long-term temperature trends  
- Analyze seasonal climate variations  
- Detect extreme climate events  
- Compare climatic behavior across cities  
- Ensure data quality and integrity  

---

##  Cities Analyzed

- Bangalore  
- Chennai  
- Delhi  
- Lucknow  
- Mumbai  
- Bhubaneswar  
- Rourkela  

---

##  Data Integrity Note

The dataset labeled **Rajasthan (Jodhpur)** was found to be an exact duplicate of the Bangalore dataset (verified via hash comparison).  
It was removed to avoid biased analysis.

---

##  Techniques Used

### Trend Analysis
- Mann-Kendall Trend Test  
- Sen’s Slope Estimation  
- Ordinary Least Squares (OLS) Regression  

### Climate Metrics
- Annual Mean Temperature  
- Temperature Anomalies  
- Seasonal Trends  
- Tmin vs Tmax Asymmetry  

### Extreme Event Analysis
- Heatwaves (IMD definition: ≥40°C for ≥3 days)  
- Warm Nights (TN90p index)  
- Extreme Rainfall Events  
- Consecutive Wet Days (CWD)  

---

##  Project Structure
Climate_Analysis/
│
├── data/ # Raw datasets
├── scripts/ # Analysis scripts
│ ├── 01_data_loading.py
│ ├── 02_data_cleaning.py
│ ├── 03_descriptive_analysis.py
│ ├── 04_trend_analysis.py
│ ├── 05_extreme_events.py
│ ├── 06_visualizations.py
│
├── results/ # Generated figures and outputs
│
├── README.md
├── requirements.txt


---

##  Key Findings

- All cities exhibit a warming trend  
- Mumbai shows the highest warming rate (~+0.47°C/decade)  
- Nighttime temperatures are increasing faster than daytime temperatures  
- Heatwave frequency has increased in northern cities (Delhi, Lucknow)  
- Rainfall patterns show high variability rather than consistent trends  
- Temperature anomalies show a clear shift toward warmer conditions in recent years  

---

##  Visualizations

All figures are available in the `results/` folder, including:

- Annual temperature trends  
- Seasonal warming heatmaps  
- Temperature anomaly heatmaps  
- Rainfall time series  
- Heatwave event timelines  
- Warm night trends  

---

##  How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Run scripts in order:

python scripts/01_data_loading.py
python scripts/02_data_cleaning.py
python scripts/03_descriptive_analysis.py
python scripts/04_trend_analysis.py
python scripts/05_extreme_events.py
python scripts/06_visualizations.py

## Significance

This project demonstrates how data analysis can transform raw climate data into meaningful insights, enabling better understanding of climate change patterns and associated risks.
