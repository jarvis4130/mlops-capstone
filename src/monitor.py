import sys
import os
sys.path.insert(0, 'src')

import pandas as pd
from sklearn.model_selection import train_test_split
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from preprocess import load_data, remove_outliers, FEATURES, TARGET

os.makedirs("reports", exist_ok=True)

df = remove_outliers(load_data())
ref, curr = train_test_split(df[FEATURES + [TARGET]], test_size=0.3, random_state=42)

report = Report([
    DataDriftPreset(),
    DataSummaryPreset(),
])
result = report.run(reference_data=ref, current_data=curr)
result.save_html("reports/drift_report.html")
print("Report saved to reports/drift_report.html")
