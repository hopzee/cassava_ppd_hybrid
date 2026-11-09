# Manual testing checklist

- [ ] Use Python 3.11.
- [ ] Install dependencies with `pip install -r requirements.txt`.
- [ ] Start the app with `streamlit run app.py`.
- [ ] Select synthetic demo data and train the model to test the complete workflow.
- [ ] Confirm the app shows a chronological 70/15/15 batch split.
- [ ] Confirm Bi-LSTM and persistence forecasting MAE/RMSE are displayed.
- [ ] Confirm XGBoost and Random Forest classification metrics are displayed.
- [ ] Confirm the app warns that synthetic metrics are not real-world validation.
- [ ] Estimate a batch DRI and inspect/download its 72-hour environmental forecast.
- [ ] Upload real environmental and cassava outcome CSV files using the required schema.
- [ ] Confirm mismatched batch IDs, missing columns, invalid labels, missing harvest timestamps and incomplete time windows are rejected clearly.
- [ ] Confirm each real batch has 24 pre-harvest hourly observations and 72 post-harvest hourly observations, all linked to its harvest timestamp.

A successful synthetic run only verifies that the software workflow can execute. It does not validate the scientific accuracy of the model.
