import os
import numpy as np
import pandas as pd

from loguru import logger


def sample_by_age_sex(df, n_max=700, seed=42):
    """
    Deterministically (with seed) sample patients based on Age Group and Sex.
    For each (Age Group, Sex) group, return min(n_max, 75% of group size), shuffled with seed.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'Age Group', 'Sex', and 'Patient ID' columns.
        n_max (int): Max number of samples per group. Default is 700.
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: Sampled DataFrame.
    """
    # Shuffle dataframe with the given seed
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Group by Age Group and Sex
    grouped = df_shuffled.groupby(['Age Group', 'Sex'])

    # Sampling function
    def custom_sample(group):
        n = min(n_max, int(0.75 * len(group)))
        return group.head(n)

    # Apply sampling
    sampled_df = grouped.apply(custom_sample).reset_index(drop=True)

    return sampled_df


if __name__ == "__main__":
    SEED = 42
    TRAIN_SAMPLE_SIZE = 700  # size for each sex-age group [male:20-30, male-30-40, ...]
    TEST_SAMPLE_SIZE = 200  # size for each sex-age group

    logger.info("Starting data sampling...")

    all_patient_info = pd.read_csv("../data/metadata/patient_information_113528_19-9-2024.csv").rename(columns={"Study ID": "study_id", "Age at Scan time": "Age"})
    normality_data = pd.read_csv("../data/metadata/report_42462_normal_heart_great_vessels.csv")

    # --- filter only the studies with normal heart and great vessels
    normality_data = normality_data[normality_data["isnormal"] == "Yes"]
    normal_studies = list(normality_data["study_id"])

    # --- normal data should overlap with patient info
    patient_info = all_patient_info[all_patient_info["study_id"].isin(normal_studies)]

    # --- research consent should be True
    patient_info = patient_info[patient_info["Research Consent"] == "1"]

    # --- remove na values
    patient_info.dropna(subset=['Age', 'Height', 'Sex', 'Weight', 'Ethnicity'], inplace=True)
    patient_info.reset_index(drop=True, inplace=True)

    # --- filter based on heght and weight
    patient_info = patient_info[patient_info["Height"] < 2.3]
    patient_info = patient_info[patient_info["Height"] > 1.0]

    patient_info = patient_info[patient_info["Weight"] < 180]
    patient_info = patient_info[patient_info["Weight"] > 30]

    # ignore ages < 20 and ages >= 80
    patient_info = patient_info[patient_info["Age"] >= 20]
    patient_info = patient_info[patient_info["Age"] < 80]


    # --- adding the age group
    bins = list(range(10, 90, 10)) + [np.inf]
    labels = [f"{i}-{i+9}" for i in bins[:-2]] + ['80+']
    patient_info['Age Group'] = pd.cut(patient_info['Age'], bins=bins, labels=labels, right=False)

    # --- print some output
    logger.info("Number of all filtered patients:" + str(len(patient_info)) + "\n")
    logger.info("Patient info head:\n" + patient_info.head().to_string() + "\n")
    logger.info("Subgroup counts:\n" + patient_info.groupby(["Age Group", "Sex"]).size().to_string() + "\n")

    # --- sample training and test data
    train_sample_path = "../data/samples/train_data.csv"
    sample_train = sample_by_age_sex(patient_info, n_max=TRAIN_SAMPLE_SIZE, seed=SEED)
    sample_train.to_csv(train_sample_path, index=False)
    logger.success(f"Train data sampled successfully to {train_sample_path}.")

    test_sample_path = "../data/samples/test_data.csv"
    df_train_excluded = patient_info[~patient_info["study_id"].isin(sample_train["study_id"])]
    sample_test = sample_by_age_sex(df_train_excluded, n_max=TEST_SAMPLE_SIZE, seed=SEED)
    sample_test.to_csv(test_sample_path, index=False)
    logger.success(f"Test data sampled successfully to {test_sample_path}.")
