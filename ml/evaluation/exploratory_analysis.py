
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "ml" / "data" / "raw" / "Dataset Clean.xlsx"
)

EDA_DIR = ROOT / "reports" / "eda"
STAT_DIR = ROOT / "reports" / "statistical"

EDA_DIR.mkdir(parents=True, exist_ok=True)
STAT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = (
    "Output(failed implantation or successful implantation)"
)

# The numerical encoding is observed, but its clinical
# interpretation must be verified against source documentation.
TARGET_LABELS = {
    0: "Class 0",
    1: "Class 1"
}

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

NUMERIC_FEATURES = [
    "Female Age",
    "AMH",
    "Male Age",
    "Sperm count",
    "Sperm motility",
    "Sperm morphology",
    "Oocyte Maturity MII",
    "Oocyte Maturity MI",
    "Oocyte Maturity others",
    "Number of fertilized eggs",
    "Embryo Grade 1",
    "Embryo Grade 2",
    "Embryo Grade 3",
    "Embryo Grade 4",
    "Number of embryos",
    "frezzing"
]

EMBRYO_FEATURES = [
    "Embryo Grade 1",
    "Embryo Grade 2",
    "Embryo Grade 3",
    "Embryo Grade 4"
]

# Excluded from pre-transfer analyses:
# - Beta-HCG: potentially measured after treatment.
# - Medicines and treatments: numerical coding not yet verified.
# - Egg retrieval date: temporal variable requiring
#   documented interpretation.


# ============================================================
# 3. DATA LOADING
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_excel(DATA_PATH)

    df.columns = (
        df.columns.astype(str).str.strip()
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column not found: {TARGET}"
        )

    for column in NUMERIC_FEATURES:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    print("=" * 65)
    print("FERTILITYIQ - EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    print("Dataset shape:", df.shape)
    print("Missing cells:", df.isna().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    return df


# ============================================================
# 4. GENERAL STATISTICAL SUMMARY
# ============================================================

def generate_summary(df):

    features = [
        col for col in NUMERIC_FEATURES
        if col in df.columns
    ]

    summary = df[features].describe().T

    summary["missing_count"] = (
        df[features].isna().sum()
    )

    summary["missing_percentage"] = (
        df[features].isna().mean() * 100
    ).round(2)

    summary.to_csv(
        STAT_DIR / "descriptive_statistics.csv"
    )

    print("\nDESCRIPTIVE STATISTICS")
    print(summary.to_string())

    return summary


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

def analyze_target(df):

    target_data = df[TARGET].dropna()

    invalid = ~target_data.isin([0, 1])

    if invalid.any():
        raise ValueError(
            "Unexpected target values found: "
            f"{target_data[invalid].unique().tolist()}"
        )

    counts = (
        target_data.value_counts()
        .reindex([0, 1], fill_value=0)
    )

    percentages = (
        target_data.value_counts(normalize=True)
        .reindex([0, 1], fill_value=0) * 100
    )

    result = pd.DataFrame({
        "class": [0, 1],
        "label": [
            TARGET_LABELS[0],
            TARGET_LABELS[1]
        ],
        "count": counts.values,
        "percentage": percentages.values
    })

    result.to_csv(
        STAT_DIR / "target_distribution.csv",
        index=False
    )

    plt.figure()

    ax = sns.barplot(
        data=result,
        x="label",
        y="count"
    )

    ax.set_title("Recorded Implantation Outcome Distribution")
    ax.set_xlabel("Encoded outcome class")
    ax.set_ylabel("Number of records")

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "target_distribution.png",
        dpi=300
    )

    plt.close()

    print("\nTARGET DISTRIBUTION")
    print(result.to_string(index=False))

    return result


# ============================================================
# 6. UNIVARIATE ANALYSIS
# ============================================================

def analyze_numeric_distributions(df):

    for column in NUMERIC_FEATURES:

        if column not in df.columns:
            continue

        values = df[column].dropna()

        if values.empty:
            continue

        fig, axes = plt.subplots(
            1, 2, figsize=(13, 5)
        )

        sns.histplot(
            values,
            kde=True,
            ax=axes[0]
        )

        axes[0].set_title(
            f"{column} - Distribution"
        )

        sns.boxplot(
            x=values,
            ax=axes[1]
        )

        axes[1].set_title(
            f"{column} - Boxplot"
        )

        plt.tight_layout()

        filename = (
            column.lower()
            .replace(" ", "_")
            .replace("%", "percentage")
        )

        plt.savefig(
            EDA_DIR / f"{filename}_analysis.png",
            dpi=300
        )

        plt.close(fig)


# ============================================================
# 7. AMH EXTREME-VALUE INVESTIGATION
# ============================================================

def investigate_amh(df):

    if "AMH" not in df.columns:
        return

    values = df["AMH"].dropna()

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    potential_outliers = df[
        (df["AMH"] < lower) |
        (df["AMH"] > upper)
    ]

    report = {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "potential_outlier_count": int(
            len(potential_outliers)
        ),
        "iqr_lower_fence": float(lower),
        "iqr_upper_fence": float(upper)
    }

    (STAT_DIR / "amh_outlier_analysis.json").write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    potential_outliers[
        ["AMH", TARGET]
    ].to_csv(
        STAT_DIR / "amh_potential_outliers.csv",
        index=False
    )

    print("\nAMH INVESTIGATION")
    print(json.dumps(report, indent=4))


# ============================================================
# 8. EMBRYO GRADE ANALYSIS
# ============================================================

def analyze_embryo_grades(df):

    available = [
        col for col in EMBRYO_FEATURES
        if col in df.columns
    ]

    if not available:
        return

    grade_summary = df[available].agg(
        ["mean", "median", "min", "max", "sum"]
    ).T

    grade_summary.to_csv(
        STAT_DIR / "embryo_grade_summary.csv"
    )

    plt.figure(figsize=(10, 6))

    totals = df[available].sum()

    sns.barplot(
        x=totals.index,
        y=totals.values
    )

    plt.title("Recorded Embryo Grade Counts")
    plt.xlabel("Embryo grade category")
    plt.ylabel("Total recorded count")

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "embryo_grade_distribution.png",
        dpi=300
    )

    plt.close()

    print("\nEMBRYO GRADE SUMMARY")
    print(grade_summary.to_string())


# ============================================================
# 9. SPEARMAN CORRELATION ANALYSIS
# ============================================================

def analyze_correlations(df):

    features = [
        col for col in NUMERIC_FEATURES
        if col in df.columns
    ]

    numeric = df[features].copy()

    correlation = numeric.corr(
        method="spearman"
    )

    correlation.to_csv(
        STAT_DIR / "spearman_correlation.csv"
    )

    plt.figure(figsize=(15, 12))

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f"
    )

    plt.title(
        "Spearman Correlation - Clinical and Embryological Features"
    )

    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "spearman_correlation.png",
        dpi=300
    )

    plt.close()

    # Correlation against the target is calculated separately.
    target_corr = []

    valid = df[df[TARGET].isin([0, 1])]

    for column in features:

        pair = valid[[column, TARGET]].dropna()

        if len(pair) < 3:
            continue

        if pair[column].nunique() < 2:
            continue

        result = stats.spearmanr(
            pair[column],
            pair[TARGET]
        )

        target_corr.append({
            "feature": column,
            "sample_size": len(pair),
            "spearman_r": result.statistic,
            "p_value": result.pvalue
        })

    results = pd.DataFrame(target_corr)

    if not results.empty:

        results["adjusted_p_value"] = multipletests(
            results["p_value"],
            method="fdr_bh"
        )[1]

        results.to_csv(
            STAT_DIR / "target_correlation_analysis.csv",
            index=False
        )

    print("\nTARGET CORRELATION ANALYSIS")
    print(results.to_string(index=False))


# ============================================================
# 10. OUTCOME SUBGROUP COMPARISONS
# ============================================================

def compare_outcome_groups(df):

    valid = df[df[TARGET].isin([0, 1])].copy()

    results = []

    for column in NUMERIC_FEATURES:

        if column not in valid.columns:
            continue

        group0 = valid.loc[
            valid[TARGET] == 0,
            column
        ].dropna()

        group1 = valid.loc[
            valid[TARGET] == 1,
            column
        ].dropna()

        if len(group0) < 3 or len(group1) < 3:
            continue

        if group0.nunique() < 2 or group1.nunique() < 2:
            continue

        test = stats.mannwhitneyu(
            group0,
            group1,
            alternative="two-sided"
        )

        results.append({
            "feature": column,
            "class_0_n": len(group0),
            "class_1_n": len(group1),
            "class_0_median": group0.median(),
            "class_1_median": group1.median(),
            "class_0_mean": group0.mean(),
            "class_1_mean": group1.mean(),
            "mann_whitney_u": test.statistic,
            "p_value": test.pvalue
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:

        result_df["adjusted_p_value"] = multipletests(
            result_df["p_value"],
            method="fdr_bh"
        )[1]

        result_df["significant_at_0_05"] = (
            result_df["adjusted_p_value"] < 0.05
        )

        result_df.to_csv(
            STAT_DIR / "outcome_group_comparisons.csv",
            index=False
        )

    print("\nOUTCOME GROUP COMPARISONS")
    print(result_df.to_string(index=False))


# ============================================================
# 11. MAIN PIPELINE
# ============================================================

def main():

    df = load_data()

    print("\nStarting research analysis...")

    generate_summary(df)

    analyze_target(df)

    analyze_numeric_distributions(df)

    investigate_amh(df)

    analyze_embryo_grades(df)

    analyze_correlations(df)

    compare_outcome_groups(df)

    print("\n" + "=" * 65)
    print("EDA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("Visualization directory:", EDA_DIR)

    print("Statistical reports:", STAT_DIR)


if __name__ == "__main__":
    main()
