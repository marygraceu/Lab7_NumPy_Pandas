import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# STUDENT CONFIGURATION
# ==========================================

STUDENT_ID = "TUPM-26-1140"
SEED_DIGIT = 0
ID_SUM = 14

# ==========================================
# LOAD SECOM MEASUREMENT DATA
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = PROJECT_DIR / "data" / "secom.data"

data = np.genfromtxt(DATA_PATH)

X = pd.DataFrame(
    data,
    columns=[
        f"sensor_{i+1:03d}"
        for i in range(data.shape[1])
    ]
)

print("SECOM DATASET LOADED")
print("Dataset Shape:", X.shape)

# ==========================================
# 1. IDENTIFY MISSING AND CONSTANT FEATURES
# ==========================================

missing_features = X.columns[
    X.isna().any()
].tolist()

constant_features = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

print("\n===== FEATURE IDENTIFICATION =====")

print("Features with Missing Values:")
print(missing_features)

print("\nFeatures with Constant Values:")
print(constant_features)

print("Number of Missing Features:", len(missing_features))
print("Number of Constant Features:", len(constant_features))

# ==========================================
# 2. SELECT USABLE FEATURES REPRODUCIBLY
# ==========================================

usable_features = [
    col for col in X.columns
    if col not in constant_features
]

rng = np.random.default_rng(SEED_DIGIT)

number_selected = 5 + (SEED_DIGIT % 3)

selected_features = rng.choice(
    usable_features,
    size=number_selected,
    replace=False
).tolist()

selected_data = X[selected_features]

print("\n===== SELECTED FEATURES =====")
print(selected_features)
print("Number of Selected Features:", len(selected_features))

# ==========================================
# 3. FEATURE-WISE STATISTICS
# ==========================================

feature_means = selected_data.mean()
feature_medians = selected_data.median()
feature_std = selected_data.std(ddof=1)

missing_percentage = (
    selected_data.isna().mean() * 100
)

print("\n===== FEATURE-WISE STATISTICS =====")

print("\nMeans:")
print(feature_means)

print("\nMedians:")
print(feature_medians)

print("\nSample Standard Deviations:")
print(feature_std)

print("\nMissing Percentages:")
print(missing_percentage)

# ==========================================
# 4. ROBUST STANDARDIZATION USING MAD
# ==========================================

mad = selected_data.sub(
    feature_medians
).abs().median()

safe_mad = mad.replace(0, np.nan)

robust_z = (
    0.6745 *
    selected_data.sub(feature_medians).div(safe_mad)
)

robust_z = robust_z.replace(
    [np.inf, -np.inf],
    np.nan
)

print("\n===== MEDIAN ABSOLUTE DEVIATION =====")
print(mad)

# ==========================================
# 5. IDENTIFY FLAGGED MEASUREMENTS
# ==========================================

flagged = (
    robust_z.abs() > 3.5
).fillna(False)

flagged_count = flagged.sum()

valid_count = selected_data.notna().sum()

flagged_percentage = (
    flagged_count /
    valid_count.replace(0, np.nan)
) * 100

total_flagged = int(
    flagged.to_numpy().sum()
)

print("\n===== FLAGGED MEASUREMENTS =====")
print(flagged_count)

print("\nFlagged Percentages:")
print(flagged_percentage)

print("\nTotal Flagged Observations:", total_flagged)

# ==========================================
# 6. RANK FEATURES AND GENERATE CSV SUMMARY
# ==========================================

summary = pd.DataFrame({
    "Feature": selected_features,
    "Mean": feature_means.values,
    "Median": feature_medians.values,
    "Sample_Std": feature_std.values,
    "Missing_Percentage": missing_percentage.values,
    "MAD": mad.values,
    "Flagged_Count": flagged_count.values,
    "Flagged_Percentage": flagged_percentage.values
})

summary["Flagged_Percentage"] = (
    summary["Flagged_Percentage"].fillna(0)
)

summary = summary.sort_values(
    by="Flagged_Percentage",
    ascending=False,
    kind="stable"
).reset_index(drop=True)

summary["Rank"] = range(1, len(summary) + 1)

OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "measurement_quality.csv"

summary.to_csv(
    CSV_PATH,
    index=False
)

print("\n===== RANKED FEATURE SUMMARY =====")
print(summary.to_string(index=False))

# ==========================================
# VERIFY VECTORIZED RESULTS
# ==========================================

loop_flags = np.zeros(
    selected_data.shape,
    dtype=bool
)

for j, col in enumerate(selected_features):

    median_value = selected_data[col].median()

    mad_value = (
        selected_data[col] - median_value
    ).abs().median()

    if pd.notna(mad_value) and mad_value != 0:

        z = (
            0.6745 *
            (selected_data[col] - median_value)
            / mad_value
        )

        loop_flags[:, j] = (
            z.abs() > 3.5
        ).fillna(False).to_numpy()

vectorized_verified = np.array_equal(
    flagged.to_numpy(),
    loop_flags
)

# ==========================================
# ASSESSMENT DATA
# ==========================================

highest_missing_feature = missing_percentage.idxmax()

highest_missing_value = missing_percentage.max()

highest_flagged_row = summary.iloc[0]

print("\n")
print("========================================")
print("             ASSESSMENT DATA")
print("========================================")

print(
    "Selected Features:",
    selected_features
)

print(
    "Highest Missing Percentage:",
    f"{highest_missing_value:.2f}% ({highest_missing_feature})"
)

print(
    "Feature with Highest Flagged Percentage:",
    f"{highest_flagged_row['Feature']} "
    f"({highest_flagged_row['Flagged_Percentage']:.2f}%)"
)

print(
    "Total Flagged Observations:",
    total_flagged
)

print(
    "Vectorized Results Verified:",
    vectorized_verified
)

print("\nRequired Output: measurement_quality.csv")
print("CSV Created:", CSV_PATH.exists())
print("CSV Location:", CSV_PATH)
print("========================================")