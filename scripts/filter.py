import json
import pickle as pkl
import numpy as np
import pandas as pd
import statsmodels.api as sm

from loguru import logger


def remove_outliers_zscore(df, column="residuals", threshold=3, stats=None):
    """
    Removes outliers from a dataframe based on the Z-score method.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column (str): The name of the column to check for outliers.
    threshold (float): The Z-score threshold to identify outliers (default is 3).

    Returns:
    pd.DataFrame: Dataframe without outliers.
    pd.DataFrame: Dataframe containing only the outliers.
    """

    # Calculate the Z-scores for the specified column
    if stats:
        mean_value = stats['mean']
        std_value = stats['std']
    else:
        mean_value = df[column].mean()
        std_value = df[column].std()
    
    df['z_score'] = (df[column] - mean_value) / std_value

    # Identify outliers
    df_cleaned = df[np.abs(df['z_score']) <= threshold]
    df_large = df[df['z_score'] > threshold]
    df_small = df[df['z_score'] < -threshold]


    # Drop the z_score column before returning
    df_cleaned = df_cleaned.drop(columns=['z_score'])
    df_large = df_large.drop(columns=['z_score'])
    df_small = df_small.drop(columns=['z_score'])

    return df_cleaned, df_large, df_small


if __name__ == "__main__":
    dataset_id = "processed_dummy_1"

    # Load regression model
    with open('../data/regression_models/lm_volume.pkl', 'rb') as f:
        model = pkl.load(f)

    # Load regression model stats
    with open(f'../data/regression_models/lm_residual_stats.json', 'r') as f:
        stats = json.load(f)
        
    df = pd.read_csv(f'../data/{dataset_id}/info/info.csv')

    X = df[['Age', 'Sex', 'Weight', 'BMI']]
    X = sm.add_constant(X)

    # filter based on volume
    df['predicted_volume'] = model.predict(X)
    df['residuals'] = df["volume"] - df['predicted_volume']

    df_clean, df_large, df_small = remove_outliers_zscore(df, column="residuals", threshold=3, stats=stats)
    df_clean.to_csv(f'../data/{dataset_id}/info/normal_heart.csv', index=False)
    df_large.to_csv(f'../data/{dataset_id}/info/large_heart.csv', index=False)
    df_small.to_csv(f'../data/{dataset_id}/info/small_heart.csv', index=False)
    logger.info("Removed outliers based on Z-score method.")