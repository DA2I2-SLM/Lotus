import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt


def _to_error(metric_value: float) -> float:
    # Some AutoGluon metrics are returned in "higher is better" form.
    return -metric_value if metric_value < 0 else metric_value


def evaluate_over_windows(predictor, test_data, prediction_length: int, num_windows: int) -> pd.DataFrame:
    rows = []
    for w in range(1, num_windows + 1):
        cutoff = -(w * prediction_length)
        scores = predictor.evaluate(test_data, metrics=["MSE", "MAE", "MASE"], cutoff=cutoff)
        rows.append(
            {
                "window": w,
                "MSE": _to_error(float(scores["MSE"])),
                "MAE": _to_error(float(scores["MAE"])),
                "MASE": _to_error(float(scores["MASE"])),
            }
        )
    return pd.DataFrame(rows)


data = TimeSeriesDataFrame.from_path(
    "https://autogluon.s3.amazonaws.com/datasets/timeseries/australian_electricity_subset/test.csv"
)

num_test_windows = 3
prediction_length = 48
train_data, test_data = data.train_test_split(num_test_windows * prediction_length)

predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data,
    presets="chronos2",
)

predictions = predictor.predict(train_data)
predictions.head()


# Generate predictions for multiple windows
predictions_per_window = predictor.backtest_predictions(test_data, num_val_windows=num_test_windows)

metrics_df = evaluate_over_windows(
    predictor=predictor,
    test_data=test_data,
    prediction_length=prediction_length,
    num_windows=num_test_windows,
)
print("\nPer-window metrics:")
print(metrics_df.to_string(index=False))
print("\nMean metrics across windows:")
print(
    metrics_df[["MSE", "MAE", "MASE"]]
    .mean()
    .to_string(float_format=lambda x: f"{x:.6f}")
)
print("\nStd metrics across windows:")
print(
    metrics_df[["MSE", "MAE", "MASE"]]
    .std()
    .to_string(float_format=lambda x: f"{x:.6f}")
)

# Plot predictions for the first two time series
item_ids = test_data.item_ids[:2].tolist()
all_predictions = pd.concat(predictions_per_window)
predictor.plot(test_data, all_predictions, max_history_length=300, item_ids=item_ids)

# Optional: Plot the cutoff dates with dashed vertical lines
for cutoff in range(-num_test_windows * prediction_length, 0, prediction_length):
    for i, ax in enumerate(plt.gcf().axes):
        cutoff_timestamp = test_data.loc[item_ids[i]].index[cutoff]
        ax.axvline(cutoff_timestamp, color='gray', linestyle='--')
