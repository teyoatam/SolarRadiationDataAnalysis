import streamlit as st # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import fetch_data, process_data, visual_data # type: ignore
import os

def visualize_data(df):
    """
    Visualize data including line charts, summary statistics, correlation analysis, and histograms.

    Args:
    df (pandas.DataFrame): Processed DataFrame.
    """
    # Add section headers
    st.header('Data Visualization')

    # Add interactive elements
    selected_variable = st.selectbox('Select Variable', ['GHI', 'DNI', 'DHI'])

    # Generate visualization based on selected variable
    if selected_variable == 'GHI':
        st.line_chart(df['GHI'])
    elif selected_variable == 'DNI':
        st.line_chart(df['DNI'])
    elif selected_variable == 'DHI':
        st.line_chart(df['DHI'])

    # Add section headers
    st.header('Summary Statistics')

    # Display summary statistics
    st.write(df.describe())

    # Add section headers
    st.header('Correlation Analysis')

    # Calculate correlation matrix
    correlation_matrix = processed_data.corr()

    # Create a new figure and axis using plt.subplots()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display correlation matrix as heatmap on the axis
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)

    # Pass the figure object to st.pyplot()
    st.pyplot(fig)

    # Add section headers
    st.header('Histograms')

    # Display histograms for selected variables
    st.bar_chart(df[selected_variable].value_counts())

# Get the current directory of the script
current_dir = os.path.dirname(__file__)

# Navigate to the data folder from the script folder
data_folder = os.path.join(current_dir, '../data')

# Fetch data
data = fetch_data(os.path.join(data_folder, 'benin-malanville.csv'))

# Process data
try:
    processed_data = process_data(data)
    visual_data(processed_data)
    visualize_data(processed_data)
except Exception as e:
    st.error(f"An error occurred: {e}")