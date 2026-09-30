import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

df = pd.read_csv("data/raw/spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "email"]

reference = df.iloc[:3000]
current = df.iloc[3000:]

report = Report(metrics=[DataDriftPreset()])

report.run(
    reference_data=reference,
    current_data=current
)

report.save_html("reports/data_drift_report.html")

print("✅ Data Drift Report Saved Successfully!")