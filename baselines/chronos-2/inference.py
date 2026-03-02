import fev
import torch
import numpy as np
from tqdm import tqdm
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

task = fev.Task(
    dataset_path="autogluon/chronos_datasets",
    dataset_config="dominick",
    horizon=8,
)

predictions_per_window = []
for window in task.iter_windows():
    past_data, future_data = window.get_input_data()
    # Get ground truth from the window
    ground_truth_data = window.get_ground_truth()
    print(f"Total series: {len(past_data)}")

    context_list = [torch.tensor(ts[task.target], dtype=torch.float32) for ts in past_data]

    batch_size = 1024
    all_medians = []

    for i in tqdm(range(0, len(context_list), batch_size), desc="Inference"):
        batch = context_list[i : i + batch_size]
        result = pipeline.predict_quantiles(
            batch,
            prediction_length=task.horizon,
            quantile_levels=[0.5],
        )

        if isinstance(result, tuple):
            quantiles = result[0]
        else:
            quantiles = result

        if isinstance(quantiles, torch.Tensor):
            quantiles = quantiles.cpu().numpy()

        quantiles = np.array(quantiles).squeeze()
        if quantiles.ndim == 1:
            quantiles = quantiles.reshape(1, -1)

        all_medians.append(quantiles)

    all_medians = np.concatenate(all_medians, axis=0)
    print(f"Final medians shape: {all_medians.shape}")

    # --- Compute MSE manually ---
    mse_per_series = []
    for i, gt_ts in enumerate(ground_truth_data):
        gt = np.array(gt_ts[task.target], dtype=np.float64)
        pred = all_medians[i]
        length = min(len(pred), len(gt))
        mse_per_series.append(np.mean((pred[:length] - gt[:length]) ** 2))

    overall_mse = np.mean(mse_per_series)
    print(f"\n=== MSE: {overall_mse:.6f} ===")
    print(f"=== RMSE: {np.sqrt(overall_mse):.6f} ===")
    print(f"=== Median per-series MSE: {np.median(mse_per_series):.6f} ===\n")

    # --- fev format for MASE ---
    predictions = [
        {"predictions": [float(v) for v in all_medians[i]]}
        for i in range(len(past_data))
    ]
    predictions_per_window.append(predictions)

summary = task.evaluation_summary(predictions_per_window, model_name="chronos-2")
print("\n=== fev Evaluation Summary ===")
print(f"MASE: {summary['MASE']:.6f}")
print(f"MSE:  {overall_mse:.6f}")
print(f"RMSE: {np.sqrt(overall_mse):.6f}")