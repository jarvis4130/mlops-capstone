import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import glob
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from preprocess import FEATURES, TARGET, load_data, remove_outliers

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def get_reference_and_current():
    df = remove_outliers(load_data())

    # reference = first 3 cities, current = last 3 cities (simulating new data)
    cities = df["City"].unique()
    reference = df[df["City"].isin(cities[:3])][FEATURES + [TARGET]]
    current = df[df["City"].isin(cities[3:])][FEATURES + [TARGET]]

    return reference, current

def run_monitoring():
    reference, current = get_reference_and_current()
    print(f"Reference size: {reference.shape}, Current size: {current.shape}")

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)

    output_path = f"{REPORTS_DIR}/drift_report.html"
    report.save_html(output_path)
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    run_monitoring()
