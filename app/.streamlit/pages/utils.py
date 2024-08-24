import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def fetch_data(file_path):
    """
    Load data from a CSV file and return a DataFrame.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    pandas.DataFrame: DataFrame containing the loaded data.
    """
    df = pd.read_csv(file_path)
    return df

def process_data(df):
    """
    Process the DataFrame by handling missing values, outliers, and incorrect entries.

    Args:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    pandas.DataFrame: Processed DataFrame.
    """
    # Check for missing values
    print("Missing values:")
    print(df.isnull().sum())

    # Handle missing values (e.g., imputation or removal)
    df.dropna(inplace=True)  # Example: remove rows with missing values

    # Drop the 'Comments' column
    df.drop(columns=['Comments'], inplace=True)

    # Z-score outlier detection
    z_scores = stats.zscore(df[['GHI', 'DNI', 'DHI']])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)  # Keeping entries where all z-scores are less than 3
    df = df[filtered_entries]

    # Removing Duplicates
    df.drop_duplicates(inplace=True)

    # Correcting Incorrect Entries (example: replacing negative values with NaN)
    df[df < 0] = np.nan

    # Convert the Timestamp column to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Set Timestamp as the index
    df.set_index('Timestamp', inplace=True)
    return df

def visual_data(df):
    """"
    Visualize various aspects of the data.

    Args:
    df (pandas.DataFrame): Processed DataFrame.
    """
    # Plot time series data for relevant variables
    df['GHI'].plot()  # Example: plot Global Horizontal Irradiance over time

    # Calculate correlation matrix
    correlation_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

    # Plot histograms for numeric variables
    df.hist(figsize=(12, 10))
    plt.show()

    # Visualize relationships between variables using scatter plots
    plt.scatter(df['Tamb'], df['GHI'])
    plt.xlabel('Ambient Temperature (°C)')
    plt.ylabel('Global Horizontal Irradiance (W/m²)')
    plt.title('Ambient Temperature vs. GHI')
    plt.show()

    # Time Series Analysis
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['DNI'], label='DNI')
    plt.plot(df.index, df['DHI'], label='DHI')
    plt.xlabel('Timestamp')
    plt.ylabel('Solar Radiation (W/m²)')
    plt.title('Solar Radiation over Time')
    plt.legend()
    plt.show()

    # Wind Analysis
    wind_data = df[['WS', 'WSgust', 'WSstdev', 'WD', 'WDstdev']]

    # Summary Statistics for Wind Data
    print(wind_data.describe())

    # Box Plot of Wind Variables
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=wind_data)
    plt.title('Box Plot of Wind Variables')
    plt.xlabel('Wind Variables')
    plt.ylabel('Values')
    plt.show()

    # Correlation Matrix of Wind Variables
    wind_correlation_matrix = wind_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(wind_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Wind Variables')
    plt.show()

    # Temperature Analysis
    temperature_data = df[['Tamb', 'TModA', 'TModB']]

    # Summary Statistics for Temperature Data
    print(temperature_data.describe())

    # Box Plot of Temperature Variables
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=temperature_data)
    plt.title('Box Plot of Temperature Variables')
    plt.xlabel('Temperature Variables')
    plt.ylabel('Values (°C)')
    plt.show()

    # Correlation Matrix of Temperature Variables
    temperature_correlation_matrix = temperature_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(temperature_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Temperature Variables')
    plt.show()

    # Scatter Plot: Ambient Temperature vs. Module Temperatures
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Tamb', y='TModA', label='Module A Temperature')
    sns.scatterplot(data=df, x='Tamb', y='TModB', label='Module B Temperature')
    plt.xlabel('Ambient Temperature (°C)')
    plt.ylabel('Module Temperatures (°C)')
    plt.title('Scatter Plot: Ambient Temperature vs. Module Temperatures')
    plt.legend()
    plt.show()

    # Box Plot of Solar Radiation Variables
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[['GHI', 'DNI', 'DHI']])
    plt.title('Box Plot of Solar Radiation Variables')
    plt.show()

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Tamb', y='GHI')
    plt.xlabel('Ambient Temperature (°C)')
    plt.ylabel('Global Horizontal Irradiance (W/m²)')
    plt.title('Scatter Plot: Ambient Temperature vs. GHI')
    plt.show()

    # Histograms for variables like GHI, DNI, DHI, WS, and temperatures
    plt.figure(figsize=(12, 8))
    df[['GHI', 'DNI', 'DHI', 'WS', 'Tamb', 'TModA', 'TModB']].hist(bins=20, figsize=(12, 8), color='skyblue', edgecolor='black', alpha=0.7)
    plt.tight_layout()
    plt.show()