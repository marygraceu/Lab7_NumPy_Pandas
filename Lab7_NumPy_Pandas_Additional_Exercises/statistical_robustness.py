from pathlib import Path
import numpy as np
import pandas as pd

# ==================================================
# STUDENT CONFIGURATION
# ==================================================

STUDENT_ID = "TUPM-26-1140"
ID_SUM = 14
SEED_DIGIT = 0

BOOTSTRAP_REPETITIONS = 2000 + (10 * SEED_DIGIT)

# ==================================================
# FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = PROJECT_DIR / "data" / "secom.data"
LABEL_PATH = PROJECT_DIR / "data" / "secom_labels.data"

OUTPUT_DIR = BASE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# LOAD SECOM DATASET
# ==================================================

data = np.genfromtxt(DATA_PATH)

feature_columns = [
    f"sensor_{i+1:03d}"
    for i in range(data.shape[1])
]

measurements = pd.DataFrame(
    data,
    columns=feature_columns
)

labels_raw = pd.read_csv(
    LABEL_PATH,
    sep=r"\s+",
    header=None
)

labels = pd.to_numeric(labels_raw.iloc[:, 0])

# ==================================================
# DATASET VALIDATION
# ==================================================

print("===== DATASET CHECK =====")

print("Measurement Records:", len(measurements))
print("Label Records:", len(labels))
print("Measurement Features:", len(feature_columns))

if len(measurements) != len(labels):
    raise ValueError(
        "Measurement and label records do not match."
    )

# ==================================================
# CREATE LABELED DATAFRAME
# ==================================================

df = measurements.copy()

df.insert(
    0,
    "record_id",
    np.arange(1, len(df) + 1)
)

df["outcome"] = labels.map({
    -1: "Pass",
    1: "Fail"
}).values

if df["outcome"].isna().any():
    raise ValueError("Invalid outcome labels detected.")

print("\n===== SECOM DATASET LOADED =====")
print("Dataset Shape:", df.shape)
print("Original Record Identifiers Preserved")

print("\n===== OUTCOME COUNTS =====")
print(df["outcome"].value_counts())

# ==================================================
# 1. REPRODUCIBLY SELECT FIVE USABLE FEATURES
# ==================================================

pass_data = df[df["outcome"] == "Pass"]
fail_data = df[df["outcome"] == "Fail"]

usable_features = []

for feature in feature_columns:

    pass_observed = pass_data[feature].notna().sum()
    fail_observed = fail_data[feature].notna().sum()

    if pass_observed >= 10 and fail_observed >= 10:
        usable_features.append(feature)

if len(usable_features) < 5:
    raise ValueError(
        "Not enough usable features with at least "
        "10 observations in each outcome group."
    )

selection_rng = np.random.default_rng(SEED_DIGIT)

selected_features = sorted(
    selection_rng.choice(
        usable_features,
        size=5,
        replace=False
    ).tolist()
)

print("\n===== SELECTED FEATURES =====")
print(selected_features)

print("\n===== OBSERVED MEASUREMENTS =====")

for feature in selected_features:

    print(
        feature,
        "| Pass:",
        pass_data[feature].notna().sum(),
        "| Fail:",
        fail_data[feature].notna().sum()
    )

# ==================================================
# 2. PREPARATION STRATEGIES
# ==================================================

def prepare_available_case(dataframe, feature):

    pass_values = dataframe.loc[
        dataframe["outcome"] == "Pass",
        feature
    ].dropna().to_numpy()

    fail_values = dataframe.loc[
        dataframe["outcome"] == "Fail",
        feature
    ].dropna().to_numpy()

    return pass_values, fail_values


def prepare_median_imputation(dataframe, feature):

    full_dataset_median = dataframe[feature].median()

    imputed_data = dataframe[feature].fillna(
        full_dataset_median
    )

    pass_values = imputed_data[
        dataframe["outcome"] == "Pass"
    ].to_numpy()

    fail_values = imputed_data[
        dataframe["outcome"] == "Fail"
    ].to_numpy()

    return pass_values, fail_values, full_dataset_median


# ==================================================
# 3-5. STRATIFIED BOOTSTRAP ANALYSIS
# ==================================================

def bootstrap_analysis(
    dataframe,
    features,
    strategy,
    repetitions,
    seed
):

    results = []

    for feature_index, feature in enumerate(features):

        # Prepare data according to strategy
        if strategy == "Available Case":

            pass_values, fail_values = (
                prepare_available_case(
                    dataframe,
                    feature
                )
            )

            median_used = np.nan

        elif strategy == "Median Imputation":

            (
                pass_values,
                fail_values,
                median_used
            ) = prepare_median_imputation(
                dataframe,
                feature
            )

        else:
            raise ValueError("Unknown preparation strategy.")

        if len(pass_values) < 10 or len(fail_values) < 10:
            raise ValueError(
                f"Insufficient observations for {feature}."
            )

        # Observed mean difference: Fail - Pass
        observed_difference = (
            np.mean(fail_values) -
            np.mean(pass_values)
        )

        # Independent, reproducible random generator
        strategy_code = (
            0 if strategy == "Available Case" else 1
        )

        rng = np.random.default_rng(
            np.random.SeedSequence([
                seed,
                strategy_code,
                feature_index
            ])
        )

        # Stratified bootstrap:
        # Resample independently within each outcome group
        # Sampling is performed with replacement.

        bootstrap_pass = rng.choice(
            pass_values,
            size=(repetitions, len(pass_values)),
            replace=True
        ).mean(axis=1)

        bootstrap_fail = rng.choice(
            fail_values,
            size=(repetitions, len(fail_values)),
            replace=True
        ).mean(axis=1)

        # Fail mean minus Pass mean
        bootstrap_differences = (
            bootstrap_fail - bootstrap_pass
        )

        # Bootstrap mean difference
        bootstrap_mean_difference = (
            np.mean(bootstrap_differences)
        )

        # Percentile-based 95% confidence interval
        ci_lower, ci_upper = np.percentile(
            bootstrap_differences,
            [2.5, 97.5]
        )

        # Direction of mean difference
        if bootstrap_mean_difference > 0:
            direction = "Positive"
        elif bootstrap_mean_difference < 0:
            direction = "Negative"
        else:
            direction = "Zero"

        # Whether confidence interval contains zero
        ci_contains_zero = (
            ci_lower <= 0 <= ci_upper
        )

        results.append({
            "Feature": feature,
            "Strategy": strategy,
            "Pass_Count": len(pass_values),
            "Fail_Count": len(fail_values),
            "Median_Used": median_used,
            "Observed_Mean_Difference": observed_difference,
            "Bootstrap_Mean_Difference": bootstrap_mean_difference,
            "CI_95_Lower": ci_lower,
            "CI_95_Upper": ci_upper,
            "Direction": direction,
            "CI_Contains_Zero": ci_contains_zero
        })

    result_df = pd.DataFrame(results)

    # Rank features by absolute bootstrap mean difference
    result_df["Feature_Rank"] = (
        result_df["Bootstrap_Mean_Difference"]
        .abs()
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    return result_df


# ==================================================
# RUN BOTH PREPARATION STRATEGIES
# ==================================================

print("\n===== BOOTSTRAP CONFIGURATION =====")

print(
    "Bootstrap Repetitions:",
    BOOTSTRAP_REPETITIONS
)

print("Random Seed:", SEED_DIGIT)

print("\n===== AVAILABLE-CASE ANALYSIS =====")

available_results = bootstrap_analysis(
    df,
    selected_features,
    "Available Case",
    BOOTSTRAP_REPETITIONS,
    SEED_DIGIT
)

print(
    available_results.to_string(index=False)
)

print("\n===== MEDIAN IMPUTATION ANALYSIS =====")

median_results = bootstrap_analysis(
    df,
    selected_features,
    "Median Imputation",
    BOOTSTRAP_REPETITIONS,
    SEED_DIGIT
)

print(
    median_results.to_string(index=False)
)

# ==================================================
# 6. COMPARE BOTH STRATEGIES
# ==================================================

comparison = available_results.merge(
    median_results,
    on="Feature",
    suffixes=("_Available", "_Median")
)

# Compare direction
comparison["Direction_Changed"] = (
    comparison["Direction_Available"]
    != comparison["Direction_Median"]
)

# Compare feature ranking
comparison["Ranking_Changed"] = (
    comparison["Feature_Rank_Available"]
    != comparison["Feature_Rank_Median"]
)

# Compare confidence interval boundaries
comparison["CI_Lower_Change"] = (
    comparison["CI_95_Lower_Median"]
    - comparison["CI_95_Lower_Available"]
)

comparison["CI_Upper_Change"] = (
    comparison["CI_95_Upper_Median"]
    - comparison["CI_95_Upper_Available"]
)

# Absolute change in bootstrap mean difference
comparison["Absolute_Change_In_Mean_Difference"] = (
    comparison["Bootstrap_Mean_Difference_Median"]
    - comparison["Bootstrap_Mean_Difference_Available"]
).abs()

# Confidence interval interpretation
comparison["CI_Status_Changed"] = (
    comparison["CI_Contains_Zero_Available"]
    != comparison["CI_Contains_Zero_Median"]
)

# ==================================================
# ASSESSMENT DATA CALCULATIONS
# ==================================================

largest_change = comparison[
    "Absolute_Change_In_Mean_Difference"
].max()

changed_sign_features = comparison.loc[
    comparison["Direction_Changed"],
    "Feature"
].tolist()

changed_ranking_features = comparison.loc[
    comparison["Ranking_Changed"],
    "Feature"
].tolist()

reproducibility_verified = False

# ==================================================
# 7. VERIFY REPRODUCIBILITY
# ==================================================

print("\n===== REPRODUCIBILITY VERIFICATION =====")

available_repeat = bootstrap_analysis(
    df,
    selected_features,
    "Available Case",
    BOOTSTRAP_REPETITIONS,
    SEED_DIGIT
)

median_repeat = bootstrap_analysis(
    df,
    selected_features,
    "Median Imputation",
    BOOTSTRAP_REPETITIONS,
    SEED_DIGIT
)

available_reproducible = np.allclose(
    available_results[
        [
            "Observed_Mean_Difference",
            "Bootstrap_Mean_Difference",
            "CI_95_Lower",
            "CI_95_Upper"
        ]
    ].to_numpy(),
    available_repeat[
        [
            "Observed_Mean_Difference",
            "Bootstrap_Mean_Difference",
            "CI_95_Lower",
            "CI_95_Upper"
        ]
    ].to_numpy(),
    rtol=1e-12,
    atol=1e-12,
    equal_nan=True
)

median_reproducible = np.allclose(
    median_results[
        [
            "Observed_Mean_Difference",
            "Bootstrap_Mean_Difference",
            "CI_95_Lower",
            "CI_95_Upper"
        ]
    ].to_numpy(),
    median_repeat[
        [
            "Observed_Mean_Difference",
            "Bootstrap_Mean_Difference",
            "CI_95_Lower",
            "CI_95_Upper"
        ]
    ].to_numpy(),
    rtol=1e-12,
    atol=1e-12,
    equal_nan=True
)

reproducibility_verified = (
    available_reproducible
    and median_reproducible
)

print(
    "Available-Case Reproducible:",
    available_reproducible
)

print(
    "Median-Imputation Reproducible:",
    median_reproducible
)

print(
    "Reproducibility Verified:",
    reproducibility_verified
)

# ==================================================
# EXPORT STATISTICAL COMPARISON CSV
# ==================================================

comparison.to_csv(
    OUTPUT_DIR / "robustness_comparison.csv",
    index=False
)

# ==================================================
# GENERATE INTERPRETATION REPORT
# ==================================================

report_lines = []

report_lines.append(
    "STATISTICAL ROBUSTNESS AND SENSITIVITY ANALYSIS"
)

report_lines.append("=" * 55)

report_lines.append(
    f"Student ID: {STUDENT_ID}"
)

report_lines.append(
    f"Selected Features: {', '.join(selected_features)}"
)

report_lines.append(
    f"Bootstrap Repetitions: {BOOTSTRAP_REPETITIONS}"
)

report_lines.append(
    f"Random Seed: {SEED_DIGIT}"
)

report_lines.append("")

report_lines.append("STRATEGY COMPARISON")
report_lines.append("-" * 55)

for _, row in comparison.iterrows():

    feature = row["Feature"]

    report_lines.append(f"\nFeature: {feature}")

    report_lines.append(
        "Available-Case Bootstrap Mean Difference: "
        f"{row['Bootstrap_Mean_Difference_Available']:.8f}"
    )

    report_lines.append(
        "Available-Case 95% CI: "
        f"[{row['CI_95_Lower_Available']:.8f}, "
        f"{row['CI_95_Upper_Available']:.8f}]"
    )

    report_lines.append(
        "Median-Imputation Bootstrap Mean Difference: "
        f"{row['Bootstrap_Mean_Difference_Median']:.8f}"
    )

    report_lines.append(
        "Median-Imputation 95% CI: "
        f"[{row['CI_95_Lower_Median']:.8f}, "
        f"{row['CI_95_Upper_Median']:.8f}]"
    )

    report_lines.append(
        f"Direction Changed: {row['Direction_Changed']}"
    )

    report_lines.append(
        f"Ranking Changed: {row['Ranking_Changed']}"
    )

    report_lines.append(
        f"Confidence Interval Status Changed: "
        f"{row['CI_Status_Changed']}"
    )

report_lines.append("\nOVERALL INTERPRETATION")
report_lines.append("-" * 55)

report_lines.append(
    f"Largest Change in Mean Difference: {largest_change:.8f}"
)

report_lines.append(
    f"Features with Changed Difference Sign: "
    f"{len(changed_sign_features)}"
)

report_lines.append(
    f"Changed Sign Features: "
    f"{', '.join(changed_sign_features) if changed_sign_features else 'None'}"
)

report_lines.append(
    f"Features with Changed Ranking: "
    f"{len(changed_ranking_features)}"
)

report_lines.append(
    f"Changed Ranking Features: "
    f"{', '.join(changed_ranking_features) if changed_ranking_features else 'None'}"
)

report_lines.append(
    f"Reproducibility Verified: {reproducibility_verified}"
)

report_lines.append("")

report_lines.append(
    "Interpretation: The comparison evaluates whether the "
    "observed Fail-versus-Pass mean differences remain stable "
    "under available-case analysis and full-dataset median "
    "imputation. Confidence intervals containing zero indicate "
    "that zero difference remains plausible under the bootstrap "
    "analysis. Changes in direction, confidence interval status, "
    "or ranking indicate sensitivity to the preparation strategy."
)

report_lines.append("")

report_lines.append(
    "Note: Bootstrap confidence intervals describe sampling "
    "uncertainty under the observed outcome groups and the "
    "chosen resampling procedure. They do not establish causation."
)

report_path = OUTPUT_DIR / "sensitivity_report.txt"

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

# ==================================================
# DISPLAY FINAL COMPARISON
# ==================================================

print("\n===== ROBUSTNESS COMPARISON =====")

display_columns = [
    "Feature",
    "Bootstrap_Mean_Difference_Available",
    "CI_95_Lower_Available",
    "CI_95_Upper_Available",
    "Feature_Rank_Available",
    "Bootstrap_Mean_Difference_Median",
    "CI_95_Lower_Median",
    "CI_95_Upper_Median",
    "Feature_Rank_Median",
    "Direction_Changed",
    "Ranking_Changed"
]

print(
    comparison[display_columns].to_string(index=False)
)

# ==================================================
# FINAL ASSESSMENT DATA
# ==================================================

print("\n")
print("=" * 55)
print("                  ASSESSMENT DATA")
print("=" * 55)

print(
    "Selected Features:",
    len(selected_features),
    selected_features
)

print(
    "Bootstrap Repetitions:",
    BOOTSTRAP_REPETITIONS
)

print(
    "Largest Change in Mean Difference:",
    f"{largest_change:.8f}"
)

print(
    "Features with Changed Difference Sign:",
    len(changed_sign_features),
    changed_sign_features
)

print(
    "Features with Changed Ranking:",
    len(changed_ranking_features),
    changed_ranking_features
)

print(
    "Reproducibility Verified:",
    reproducibility_verified
)

# ==================================================
# REQUIRED OUTPUT VERIFICATION
# ==================================================

comparison_path = (
    OUTPUT_DIR / "robustness_comparison.csv"
)

print("\n===== REQUIRED OUTPUTS =====")

print(
    "robustness_comparison.csv Created:",
    comparison_path.exists()
)

print(
    "sensitivity_report.txt Created:",
    report_path.exists()
)

print("\nOutput Folder:", OUTPUT_DIR)

print("\n===== STATISTICAL ROBUSTNESS ANALYSIS COMPLETED =====")
