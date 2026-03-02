import pandas as pd
from chronos import BaseChronosPipeline, Chronos2Pipeline

data_dir = "/scratch/s223540177/OpenTSLM/data/test_chronos_data"

pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# # Example: M4 Hourly — one of the original Chronos pretrained datasets
# context_df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")

# pred_df = pipeline.predict_df(context_df, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])

# load dominik subset. currently it is in parquet format, we need to convert it to csv format for chronos pipeline
dataset_path = "/scratch/Projects/rl_timeseries/pretraining_data/multivariate/chronos_datasets/taxi_1h"
context_df = pd.read_parquet(f"{dataset_path}/train-00000-of-00001.parquet")

# check the columns of the dataframe, and rename them to match the expected format of chronos pipeline
print(context_df.columns)

# change the column name from id -> item_id
context_df = context_df.rename(columns={"id": "item_id"})

# convert to csv format, on the fly, without saving to disk
context_df.to_csv(f"{data_dir}/train.csv", index=False)
print(context_df.head())
quit()

pred_df = pipeline.predict_df(context_df, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])

print(pred_df.head())