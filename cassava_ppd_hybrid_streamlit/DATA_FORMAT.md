# CSV data format

Use two CSV files with the same `batch_id` values. Each batch must have a known harvest timestamp and time-aligned environmental records.

## 1. Environmental time-series file

Required columns:

```csv
batch_id,timestamp,temperature_c,relative_humidity_pct
BATCH-001,2025-01-01 00:00:00,28.4,72.0
BATCH-001,2025-01-01 01:00:00,28.1,73.2
```

For each batch, supply 24 consecutive hourly observations immediately before harvest and 72 consecutive hourly observations after harvest. The app uses the 24 pre-harvest hours as the Bi-LSTM input and the 72 post-harvest hours as its forecast target during training.

## 2. Cassava batch/outcome file

Required columns:

```csv
batch_id,variety,mechanical_damage,initial_condition,dry_matter_pct,ppd_status_72h,harvest_timestamp
BATCH-001,TME 419,1,Good,34.5,0,2025-01-02 00:00:00
```

- `mechanical_damage`: numeric score from 0 (none) to 3 (severe).
- `initial_condition`: recorded initial root condition, such as Good, Fair or Poor.
- `ppd_status_72h`: 0 or 1, based on a clearly defined and independently observed deterioration outcome at the end of the 72-hour post-harvest period.
- `harvest_timestamp`: exact harvest date and time; this anchors the 24-hour input and 72-hour target windows.

These examples show column structure only. They are not sufficient training data. The two files must contain genuinely linked measurements for at least 20 independent batches (more are recommended). Do not use generated/synthetic records as real research observations.
