import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration for each planet
PLANET_CONFIG = {
    "Mars": {
        "gravity": 3.721,          # m/s²
        "mass": 6.4171e23,         # kg
        "threshold": 2.0,
        "metadata": "02.BHV",
        "header_mapping": {
            'time_column': 'time(%Y-%m-%dT%H:%M:%S.%f)',
            'rel_time_column': 'rel_time(sec)',
            'velocity_column': 'velocity(c/s)'
        },
        "velocity_unit_conversion": 1.0,  # No conversion needed if units are consistent
        "training_data_folders": [
            {"path": "./data/mars/training/data/", "label": 1},
        ],
        "test_data_folders": [
            {"path": "./data/mars/test/data/", "label": 1},
        ]
    },
    "Lunar": {
        "gravity": 1.62,            # m/s²
        "mass": 7.342e22,           # kg
        "threshold": 2.0,
        "metadata": "12.00.mhz",
        "header_mapping": {
            'time_column': 'time_abs(%Y-%m-%dT%H:%M:%S.%f)',
            'rel_time_column': 'time_rel(sec)',
            'velocity_column': 'velocity(m/s)'
        },
        "velocity_unit_conversion": 100.0,  # Convert m/s to cm/s
        "training_data_folders": [
            {"path": "./data/lunar/training/data/S12_GradeA/", "label": 1},
        ],
        "test_data_folders": [
            {"path": "./data/lunar/test/data/S12_GradeB/", "label": 0},
            {"path": "./data/lunar/test/data/S15_GradeA/", "label": 1},
            {"path": "./data/lunar/test/data/S15_GradeB/", "label": 0},
            {"path": "./data/lunar/test/data/S16_GradeA/", "label": 1},
            {"path": "./data/lunar/test/data/S16_GradeB/", "label": 0}
        ]
    }
}

def load_data(filepath, label, header_mapping, velocity_unit_conversion):
    """Load seismic data from a CSV file and assign label."""
    try:
        logging.info(f"Loading data from {filepath}")
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)

            # Check for the required columns based on header_mapping
            required_columns = [
                header_mapping['time_column'],
                header_mapping['rel_time_column'],
                header_mapping['velocity_column']
            ]
            if not all(col in df.columns for col in required_columns):
                logging.warning(f"CSV file {filepath} does not contain the required headers: {required_columns}")
                return None, None

            # Rename the columns to standardize
            df.rename(columns={
                header_mapping['time_column']: 'time',
                header_mapping['rel_time_column']: 'rel_time',
                header_mapping['velocity_column']: 'amplitude'
            }, inplace=True)

            # Parse the 'time' column to datetime
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
            if df['time'].isnull().any():
                raise ValueError(f"Invalid datetime format in {filepath}")

            # Ensure 'amplitude' column is numeric and apply unit conversion if necessary
            df['amplitude'] = pd.to_numeric(df['amplitude'], errors='coerce') * velocity_unit_conversion
            if df['amplitude'].isnull().any():
                raise ValueError(f"Non-numeric values found in 'amplitude' column in {filepath}")

            # Log amplitude statistics
            logging.info(f"{os.path.basename(filepath)} - Amplitude Stats: min={df['amplitude'].min()}, max={df['amplitude'].max()}, mean={df['amplitude'].mean()}, std={df['amplitude'].std()}")

            # Check for anomalies
            if (df['amplitude'].abs() > 1e6).any():
                logging.warning(f"Amplitude values exceed expected range in {filepath}.")

            return df[['time', 'amplitude']], label

        else:
            logging.warning(f"Unsupported file format: {filepath}")
            return None, None

    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
        return None, None


def preprocess_data(data, planet_props):
    """Normalize amplitude based on planet's gravity and mass."""
    # Use absolute amplitude to ensure non-negative values
    data['normalized_amplitude'] = (data['amplitude'].abs() / planet_props['mass']) * planet_props['gravity']
    
    # Log normalized amplitude statistics
    logging.info(f"{planet_props['metadata']} - Normalized Amplitude Stats: min={data['normalized_amplitude'].min()}, max={data['normalized_amplitude'].max()}, mean={data['normalized_amplitude'].mean()}, std={data['normalized_amplitude'].std()}")
    
    return data

def parse_arguments():
    """Parse command-line arguments to specify the planet."""
    parser = argparse.ArgumentParser(description='Seismic Event Detection for Mars or Lunar Data')
    parser.add_argument('planet', choices=['Mars', 'Lunar'], help='Planet to process data for')
    args = parser.parse_args()
    return args


def calculate_sta_lta(data, sta_window=10, lta_window=100):
    """Calculate the Short-Term Average (STA) to Long-Term Average (LTA) ratio using normalized amplitudes."""
    # Use normalized amplitude for STA/LTA
    sta = data['normalized_amplitude'].rolling(window=sta_window, min_periods=1).mean()
    lta = data['normalized_amplitude'].rolling(window=lta_window, min_periods=1).mean()
    
    # Replace zeros in LTA to avoid division by zero
    lta.replace(0, np.nan, inplace=True)
    
    # Calculate STA/LTA ratio
    sta_lta = sta / lta
    
    # Replace NaN with 0
    sta_lta = sta_lta.fillna(0)
    
    # Log STA/LTA statistics
    logging.debug(f"STA/LTA Stats: min={sta_lta.min()}, max={sta_lta.max()}, mean={sta_lta.mean()}, std={sta_lta.std()}")
    
    # Ensure 'sta_lta' has no negative values
    if (sta_lta < 0).any():
        logging.error("Negative values detected in STA/LTA ratio, which should not occur.")
        raise ValueError("Negative STA/LTA ratios found.")
    
    return sta_lta

def detect_seismic_events(sta_lta, threshold, refractory_period=10):
    """Detect seismic events where STA/LTA ratio exceeds the threshold."""
    events = []
    last_event = -refractory_period
    for i, value in enumerate(sta_lta):
        if value > threshold and (i - last_event) > refractory_period:
            events.append(1)
            last_event = i
        else:
            events.append(0)
    return pd.Series(events, index=sta_lta.index)

def extract_features(data, planet_props):
    """Extract features from seismic data."""
    sta_lta = calculate_sta_lta(data)
    
    # Dynamic threshold based on statistical properties
    dynamic_threshold = sta_lta.mean() + (2 * sta_lta.std())
    events = detect_seismic_events(sta_lta, dynamic_threshold)
    data['sta_lta'] = sta_lta
    data['event_detected'] = events

    # Diagnostic Logging
    logging.info(f"{planet_props['metadata']} STA/LTA - min: {sta_lta.min()}, max: {sta_lta.max()}, mean: {sta_lta.mean()}, std: {sta_lta.std()}")
    logging.info(f"{planet_props['metadata']} Dynamic Threshold: {dynamic_threshold}")

    return data, events, dynamic_threshold

def main():
    pass

if __name__ == "__main__":
    main()