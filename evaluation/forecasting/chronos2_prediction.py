import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def parse_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    if any(v <= 0 for v in values):
        raise ValueError("All lengths must be positive integers.")
    return sorted(set(values))


def is_non_scalar_sequence(x) -> bool:
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        return arr.ndim > 0 and arr.size > 1
    return False


def to_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        if arr.size == 0:
            return np.nan
        if arr.size == 1:
            return arr.reshape(-1)[0]
        return np.nan
    return x


def load_long_dataframe(
    context_df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    target_column: str = "target",
) -> pd.DataFrame:
    required_cols = {id_column, timestamp_column, target_column}
    missing = required_cols - set(context_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input dataframe: {sorted(missing)}")

    base = context_df[[id_column, timestamp_column, target_column]].rename(
        columns={
            id_column: "id",
            timestamp_column: "timestamp",
            target_column: "target",
        }
    )
    base = base.explode(
        ["timestamp", "target"]
    ).reset_index(drop=True)

    # If each exploded target is still vector-valued, keep all channels by
    # expanding channel dimension into separate item ids.
    has_vector_targets = base["target"].map(is_non_scalar_sequence).any()
    if has_vector_targets:
        vectors = base["target"].apply(
            lambda x: np.asarray(x).reshape(-1)
            if isinstance(x, (list, tuple, np.ndarray))
            else np.asarray([x])
        )
        expanded = pd.DataFrame(vectors.tolist(), index=base.index)
        df = base[["id", "timestamp"]].join(expanded)
        df = df.melt(
            id_vars=["id", "timestamp"],
            var_name="channel",
            value_name="target",
        )
        df["id"] = df["id"].astype(str) + "_ch" + df["channel"].astype(int).astype(str)
        df = df.drop(columns=["channel"])
    else:
        df = base
        df["target"] = df["target"].apply(to_scalar)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["id", "timestamp", "target"])
    df["target"] = df["target"].astype("float32")
    return df


def load_context_dataframe_from_local(dataset_path: str, parquet_filename: str) -> tuple[pd.DataFrame, Path]:
    parquet_path = Path(dataset_path) / parquet_filename
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    return pd.read_parquet(parquet_path), parquet_path


def metric_to_error(raw_metric_value: float) -> float:
    # Some AutoGluon evaluate outputs are "higher is better" transformed
    # (e.g., negative MSE). MSE itself cannot be negative.
    return -raw_metric_value if raw_metric_value < 0 else raw_metric_value


def mean_metric_over_windows(predictor, test_data, pred_len: int, num_windows: int, metric_name: str):
    metrics = []
    for w in range(1, num_windows + 1):
        cutoff = -(w * pred_len)
        score = predictor.evaluate(test_data, metrics=[metric_name], cutoff=cutoff)
        raw_value = float(score[metric_name])
        metrics.append(metric_to_error(raw_value))
    return float(np.mean(metrics)), float(np.std(metrics)), metrics


def build_timeseries_dataframe(df: pd.DataFrame) -> TimeSeriesDataFrame:
    return TimeSeriesDataFrame.from_data_frame(
        df,
        id_column="id",
        timestamp_column="timestamp",
    )


def run_grid(
    df_long: pd.DataFrame,
    context_lengths: list[int],
    prediction_lengths: list[int],
    num_test_windows: int,
    seasonal_period: int,
) -> pd.DataFrame:
    rows = []
    lengths = df_long.groupby("id").size()

    for pred_len in prediction_lengths:
        for ctx_len in context_lengths:
            min_required_length = ctx_len + (num_test_windows * pred_len)
            valid_ids = lengths[lengths >= min_required_length].index
            if len(valid_ids) == 0:
                print(
                    f"[SKIP] pred_len={pred_len}, ctx_len={ctx_len}: "
                    f"no series with length >= {min_required_length}"
                )
                continue

            df_subset = df_long[df_long["id"].isin(valid_ids)].copy()
            ts_data = build_timeseries_dataframe(df_subset)
            train_data, test_data = ts_data.train_test_split(num_test_windows * pred_len)

            print(
                f"[RUN] pred_len={pred_len}, ctx_len={ctx_len}, "
                f"series={len(valid_ids)}, min_required_length={min_required_length}"
            )
            predictor = TimeSeriesPredictor(
                prediction_length=pred_len,
                eval_metric="MSE",
                eval_metric_seasonal_period=seasonal_period,
            ).fit(
                train_data,
                presets="chronos2",
                hyperparameters={"Chronos2": {"context_length": ctx_len}},
            )

            mse_mean, mse_std, mses = mean_metric_over_windows(
                predictor, test_data, pred_len, num_test_windows, "MSE"
            )
            mae_mean, mae_std, maes = mean_metric_over_windows(
                predictor, test_data, pred_len, num_test_windows, "MAE"
            )
            mase_mean, mase_std, mases = mean_metric_over_windows(
                predictor, test_data, pred_len, num_test_windows, "MASE"
            )

            rows.append(
                {
                    "context_length": ctx_len,
                    "prediction_length": pred_len,
                    "num_series": int(len(valid_ids)),
                    "mse_mean": mse_mean,
                    "mse_std": mse_std,
                    "mae_mean": mae_mean,
                    "mae_std": mae_std,
                    "mase_mean": mase_mean,
                    "mase_std": mase_std,
                    "mses_per_window": mses,
                    "maes_per_window": maes,
                    "mases_per_window": mases,
                }
            )

    if not rows:
        raise RuntimeError("No valid (context_length, prediction_length) configuration was evaluated.")
    return pd.DataFrame(rows).sort_values(["prediction_length", "context_length"])


def main():
    parser = argparse.ArgumentParser(description="Chronos2 rolling-window forecasting evaluation.")
    parser.add_argument(
        "--dataset-path",
        "--dataset_path",
        type=str,
        dest="dataset_path",
        default="/scratch/Projects/rl_timeseries/pretraining_data/multivariate/chronos_datasets/monash_australian_electricity",
        help="Local dataset directory containing parquet file.",
    )
    parser.add_argument(
        "--parquet-filename",
        type=str,
        default="train-00000-of-00001.parquet",
        help="Parquet file name under --dataset-path.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Series id column name.",
    )
    parser.add_argument(
        "--timestamp-column",
        type=str,
        default="timestamp",
        help="Timestamp column name.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="target",
        help="Target column name.",
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="336,672",
        help="Comma-separated context lengths (e.g., 336,672)",
    )
    parser.add_argument(
        "--prediction-lengths",
        type=str,
        default="48,96,336",
        help="Comma-separated prediction lengths (e.g., 48,96,336)",
    )
    parser.add_argument(
        "--num-test-windows",
        type=int,
        default=8,
        help="Number of rolling test windows per setting.",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=48,
        help="Seasonal period used by MASE (e.g., 48 for half-hourly daily seasonality).",
    )
    parser.add_argument("--grid-out", type=str, default="chronos2_mse_grid.csv")
    parser.add_argument("--pivot-out", type=str, default="chronos2_mse_pivot.csv")
    args = parser.parse_args()

    if args.num_test_windows < 2:
        raise ValueError("num_test_windows must be >= 2 for stable rolling-window evaluation.")
    if args.seasonal_period <= 0:
        raise ValueError("--seasonal-period must be a positive integer.")

    context_lengths = parse_int_list(args.context_lengths)
    prediction_lengths = parse_int_list(args.prediction_lengths)

    context_df, src = load_context_dataframe_from_local(
        dataset_path=args.dataset_path,
        parquet_filename=args.parquet_filename,
    )
    print(f"Loading dataset from local parquet: {src}")

    df_long = load_long_dataframe(
        context_df=context_df,
        id_column=args.id_column,
        timestamp_column=args.timestamp_column,
        target_column=args.target_column,
    )
    print(
        f"Prepared long dataframe with {len(df_long):,} rows and "
        f"{df_long['id'].nunique():,} series."
    )

    results = run_grid(
        df_long=df_long,
        context_lengths=context_lengths,
        prediction_lengths=prediction_lengths,
        num_test_windows=args.num_test_windows,
        seasonal_period=args.seasonal_period,
    )
    print("\nGrid Results:\n")
    print(results)

    mse_pivot = results.pivot(
        index="context_length",
        columns="prediction_length",
        values="mse_mean",
    )
    mae_pivot = results.pivot(
        index="context_length",
        columns="prediction_length",
        values="mae_mean",
    )
    mase_pivot = results.pivot(
        index="context_length",
        columns="prediction_length",
        values="mase_mean",
    )
    print("\nMean MSE (rows=context_length, cols=prediction_length):\n")
    print(mse_pivot)
    print("\nMean MAE (rows=context_length, cols=prediction_length):\n")
    print(mae_pivot)
    print("\nMean MASE (rows=context_length, cols=prediction_length):\n")
    print(mase_pivot)

    results.to_csv(args.grid_out, index=False)
    mse_pivot.to_csv(args.pivot_out)
    print(f"\nSaved grid:  {args.grid_out}")
    print(f"Saved pivot: {args.pivot_out}")


if __name__ == "__main__":
    main()
