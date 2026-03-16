# Climate Analysis Project

This project analyzes long-term climate trends across multiple Indian cities (1990–2022).

Cities included:
- Bangalore
- Chennai
- Delhi
- Lucknow
- Mumbai
- Rajasthan (Jodhpur)
- Bhubaneswar
- Rourkela

## Project Structure

data/
Raw weather datasets.

scripts/
Python scripts used for analysis.

01_data_loading.py
Loads and merges all city datasets.

02_data_cleaning.py
Performs:
- structural validation
- coverage analysis
- rainfall filtering
- anomaly detection

03_descriptive_analysis.py
Computes annual temperature statistics and trends.

## Data Cleaning Steps

1. Removed duplicate timestamps
2. Verified calendar completeness
3. Checked physical constraints:
   - tmax >= tmin
   - rainfall >= 0
4. Calculated coverage per city/year
5. Flagged statistical anomalies using Z-score
6. Fixed one corrupted temperature value

## Current Status

Cleaning complete.
Next stage: temperature trend analysis.
