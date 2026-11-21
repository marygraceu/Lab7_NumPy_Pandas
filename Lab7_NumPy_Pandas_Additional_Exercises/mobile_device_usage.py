from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================
# STUDENT CONFIGURATION
# ==========================================

STUDENT_ID = "TUPM-26-1140"
ID_SUM = 14
SEED_DIGIT = 0

# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = PROJECT_DIR / "data" / "secom.data"
LABEL_PATH = PROJECT_DIR / "data" / "secom_labels.data"

OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================
# LOAD SECOM MEASUREMENT DATA
# ==========================================

data = np.genfromtxt(DATA_PATH)

measurement_columns = [
    f"sensor_{i+1:03d}"
    for i in range(data.shape[1])
]

measurements = pd.DataFrame(
    data,
    columns=measurement_columns
)

# ==========================================
# LOAD SECOM LABEL DATA
# ==========================================

labels_raw = pd.read_csv(
    LABEL_PATH,
    sep=r"\s+",
    header=None,
    engine="python"
)

labels = pd.DataFrame({
    "outcome_code": pd.to_numeric(labels_raw.iloc[:, 0]),
    "timestamp": labels_raw.iloc[:, 1:].astype(str).agg(" ".join, axis=1)
})

# ==========================================
# CHECK DATASET ALIGNMENT
# ==========================================

print("===== DATASET CHECK =====")
print("Measurement Records:", len(measurements))
print("Label Records:", len(labels))

if len(measurements) != len(labels):
    raise ValueError("Measurement and label records do not match.")

# ==========================================
# CREATE LABELED DATAFRAME
# ==========================================

df = measurements.copy()

df.insert(
    0,
    "record_id",
    np.arange(1, len(df) + 1)
)

df["outcome_code"] = labels["outcome_code"]
df["timestamp"] = labels["timestamp"]

df["outcome"] = df["outcome_code"].map({
    -1: "Pass",
    1: "Fail"
})

if df["outcome"].isna().any():
    raise ValueError("Unexpected outcome codes found.")

print("\n===== SECOM DATASET LOADED =====")
print("Dataset Shape:", df.shape)
print("Original Record Identifiers Preserved")

# ==========================================
# OUTCOME COUNTS
# ==========================================

print("\n===== OUTCOME COUNTS =====")
print(df["outcome"].value_counts())

pass_data = df[df["outcome"] == "Pass"]
fail_data = df[df["outcome"] == "Fail"]

# ==========================================
# CREATE BALANCED SAMPLE
# ==========================================

sample_size = min(
    20 + ID_SUM,
    len(pass_data),
    len(fail_data)
)

balanced_sample = pd.concat([
    pass_data.sample(
        n=sample_size,
        random_state=SEED_DIGIT
    ),
    fail_data.sample(
        n=sample_size,
        random_state=SEED_DIGIT
    )
])

balanced_sample = balanced_sample.sort_values(
    "record_id"
).reset_index(drop=True)

print("\n===== BALANCED SAMPLE =====")
print("Pass Records:", (balanced_sample["outcome"] == "Pass").sum())
print("Fail Records:", (balanced_sample["outcome"] == "Fail").sum())
print("Balanced Sample Size:", len(balanced_sample))

# ==========================================
# SAVE BALANCED SAMPLE
# ==========================================

balanced_sample.to_csv(
    OUTPUT_DIR / "balanced_sample.csv",
    index=False
)

# ==========================================
# SELECT FIVE USABLE FEATURES
# ==========================================

selected_features = [
    "sensor_356",
    "sensor_287",
    "sensor_135",
    "sensor_155",
    "sensor_491"
]

print("\n===== SELECTED FEATURES =====")
print(selected_features)

# ==========================================
# MEDIAN IMPUTATION
# USING FULL DATASET MEDIANS
# ==========================================

full_medians = df[selected_features].median()

df[selected_features] = df[selected_features].fillna(
    full_medians
)

print("\n===== FULL DATASET MEDIANS =====")
print(full_medians)

print("\n===== MISSING VALUES AFTER IMPUTATION =====")
print(df[selected_features].isna().sum())

# Use the balanced records after full-dataset imputation
balanced_sample = df[
    df["record_id"].isin(balanced_sample["record_id"])
].copy()

balanced_sample.to_csv(
    OUTPUT_DIR / "balanced_sample.csv",
    index=False
)

# ==========================================
# GROUPED STATISTICS
# ==========================================

group_stats = balanced_sample.groupby("outcome")[
    selected_features
].agg([
    "count",
    "mean",
    "median",
    "std"
])

print("\n===== GROUPED STATISTICS =====")
print(group_stats)

# ==========================================
# EXTRACT GROUP STATISTICS
# ==========================================

mean_stats = group_stats.xs(
    "mean",
    axis=1,
    level=1
)

std_stats = group_stats.xs(
    "std",
    axis=1,
    level=1
)

count_stats = group_stats.xs(
    "count",
    axis=1,
    level=1
)

pass_means = mean_stats.loc["Pass"]
fail_means = mean_stats.loc["Fail"]

pass_stds = std_stats.loc["Pass"]
fail_stds = std_stats.loc["Fail"]

pass_counts = count_stats.loc["Pass"]
fail_counts = count_stats.loc["Fail"]

# ==========================================
# ABSOLUTE MEAN DIFFERENCE
# ==========================================

absolute_mean_difference = abs(
    pass_means - fail_means
)

# ==========================================
# POOLED WITHIN-GROUP STANDARD DEVIATION
# ==========================================

pooled_std = np.sqrt(
    (
        (pass_counts - 1) * (pass_stds ** 2)
        +
        (fail_counts - 1) * (fail_stds ** 2)
    )
    /
    (pass_counts + fail_counts - 2)
)

# ==========================================
# STANDARDIZED MEAN DIFFERENCE
# ==========================================

standardized_mean_difference = (
    absolute_mean_difference
    /
    pooled_std.replace(0, np.nan)
)

# ==========================================
# FEATURE COMPARISON
# ==========================================

feature_comparison = pd.DataFrame({
    "pass_mean": pass_means,
    "fail_mean": fail_means,
    "absolute_mean_difference": absolute_mean_difference,
    "pooled_std": pooled_std,
    "standardized_mean_difference": standardized_mean_difference
})

feature_comparison[
    "absolute_standardized_mean_difference"
] = feature_comparison[
    "standardized_mean_difference"
].abs()

# ==========================================
# RANK FEATURES
# ==========================================

feature_comparison = feature_comparison.sort_values(
    by="absolute_standardized_mean_difference",
    ascending=False
)

feature_comparison.insert(
    0,
    "rank",
    range(1, len(feature_comparison) + 1)
)

print("\n===== FEATURE COMPARISON =====")
print(feature_comparison)

# ==========================================
# NUMPY-PANDAS STATISTICS VALIDATION
# ==========================================

numpy_pandas_match = True

for outcome in ["Pass", "Fail"]:

    group = balanced_sample[
        balanced_sample["outcome"] == outcome
    ][selected_features]

    numpy_means = np.nanmean(
        group.to_numpy(),
        axis=0
    )

    pandas_means = mean_stats.loc[outcome].to_numpy()

    if not np.allclose(
        numpy_means,
        pandas_means,
        equal_nan=True
    ):
        numpy_pandas_match = False

print("\n===== STATISTICS VALIDATION =====")
print("NumPy-Pandas Statistics Match:", numpy_pandas_match)

# ==========================================
# EXPORT FEATURE COMPARISON
# ==========================================

feature_comparison.to_csv(
    OUTPUT_DIR / "feature_comparison.csv",
    index=False
)

# ==========================================
# FINAL ASSESSMENT DATA
# ==========================================

print("\n===================================")
print("          ASSESSMENT DATA")
print("===================================")

print("Balanced Sample Size:", len(balanced_sample))

print("Selected Features:", selected_features)

print(
    "Highest-Ranked Feature:",
    feature_comparison.index[0]
)

print(
    "Absolute Standardized Mean Difference:",
    feature_comparison.iloc[0][
        "absolute_standardized_mean_difference"
    ]
)

print(
    "NumPy-Pandas Statistics Match:",
    numpy_pandas_match
)

print("\n===== REQUIRED OUTPUTS =====")
print("balanced_sample.csv Created:",
      (OUTPUT_DIR / "balanced_sample.csv").exists())

print("feature_comparison.csv Created:",
      (OUTPUT_DIR / "feature_comparison.csv").exists())

print("\nCSV Locations:")
print(OUTPUT_DIR / "balanced_sample.csv")
print(OUTPUT_DIR / "feature_comparison.csv")

print("\n===== MOBILE DEVICE USAGE PROFILING COMPLETED =====")