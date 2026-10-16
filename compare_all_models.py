
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "model_comparison"

# Original Random Forest results from spatial_cv.py
models = [
    {
        "Model": "Random Forest",
        "Mean Spatial MAE": 2.7195,
        "Mean Spatial R2": -0.0939,
        "Results File": "Original spatial_cv.py output",
    }
]

# Read the saved results for each alternative model
experiments = {
    "Gradient Boosting": "gradient_boosting_spatial_cv.csv",
    "Extra Trees": "extra_trees_spatial_cv.csv",
    "SVR": "svr_spatial_cv.csv",
}

for model_name, filename in experiments.items():
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"Skipping {model_name}: file not found: {path}")
        continue

    df = pd.read_csv(path)

    models.append({
        "Model": model_name,
        "Mean Spatial MAE": df["MAE"].mean(),
        "Mean Spatial R2": df["R2"].mean(),
        "Results File": filename,
    })

summary = pd.DataFrame(models)

# Lower MAE is better; higher R2 is better.
summary = summary.sort_values(
    by="Mean Spatial MAE",
    ascending=True
).reset_index(drop=True)

summary.insert(0, "MAE Rank", range(1, len(summary) + 1))

output_file = RESULTS_DIR / "model_comparison_summary.csv"
summary.to_csv(output_file, index=False)

print("\n======================================")
print("OCEAN PLASTIC MODEL COMPARISON")
print("Spatial Cross-Validation")
print("======================================")

print(summary.to_string(
    index=False,
    formatters={
        "Mean Spatial MAE": "{:.4f}".format,
        "Mean Spatial R2": "{:.4f}".format,
    }
))

print("\nBest MAE among recorded models:",
      summary.iloc[0]["Model"])

print("\nSummary saved to:")
print(output_file)

print(
    "\nNote: Random Forest values are entered from the original "
    "spatial_cv.py output. Verify identical fold assignments "
    "before treating this as a definitive model comparison."
)
