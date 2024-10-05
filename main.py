import os
import json
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:
    
    PLANET_CONFIG = "./config.json"
    @staticmethod
    def get_planet_config(planet_name):
        with open(Config.PLANET_CONFIG, 'r') as file:
            return json.load(file).get(planet_name, None)

class DataLoader:
    def __init__(self, planet_config):
        self.gravity = planet_config['gravity']
        self.mass = planet_config['mass']
        self.header_mapping = planet_config['header_mapping']
        self.velocity_unit_conversion = planet_config.get('velocity_unit_conversion', 1.0)

    def load_data(self, filepath, label):
        """Load seismic data from a CSV file and assign label."""
        try:
            logging.info(f"Loading data from {filepath}")
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)

                # Check for the required columns based on header_mapping
                required_columns = [
                    self.header_mapping['time_column'],
                    self.header_mapping['rel_time_column'],
                    self.header_mapping['velocity_column']
                ]
                if not all(col in df.columns for col in required_columns):
                    logging.warning(f"CSV file {filepath} does not contain the required headers: {required_columns}")
                    return None, None

                # Rename the columns to standardize
                df.rename(columns={
                    self.header_mapping['time_column']: 'time',
                    self.header_mapping['rel_time_column']: 'rel_time',
                    self.header_mapping['velocity_column']: 'amplitude'
                }, inplace=True)

                # Parse the 'time' column to datetime
                df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
                if df['time'].isnull().any():
                    raise ValueError(f"Invalid datetime format in {filepath}")

                # Ensure 'amplitude' column is numeric and apply unit conversion if necessary
                df['amplitude'] = pd.to_numeric(df['amplitude'], errors='coerce') * self.velocity_unit_conversion
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


class Preprocessor:
    def __init__(self, planet_config):
        self.gravity = planet_config['gravity']
        self.mass = planet_config['mass']
        self.metadata = planet_config['metadata']

    def preprocess_data(self, data):
        """Normalize amplitude based on planet's gravity and mass."""
        # Use absolute amplitude to ensure non-negative values
        data['normalized_amplitude'] = (data['amplitude'].abs() / self.mass) * self.gravity

        # Log normalized amplitude statistics
        logging.info(f"{self.metadata} - Normalized Amplitude Stats: min={data['normalized_amplitude'].min()}, max={data['normalized_amplitude'].max()}, mean={data['normalized_amplitude'].mean()}, std={data['normalized_amplitude'].std()}")

        return data


class FeatureExtractor:
    def __init__(self, planet_config):
        self.metadata = planet_config['metadata']
        self.threshold = planet_config['threshold']

    def calculate_sta_lta(self, data, sta_window=10, lta_window=100):
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

    def detect_seismic_events(self, sta_lta, dynamic_threshold, refractory_period=10):
        """Detect seismic events where STA/LTA ratio exceeds the threshold."""
        events = []
        last_event = -refractory_period
        for i, value in enumerate(sta_lta):
            if value > dynamic_threshold and (i - last_event) > refractory_period:
                events.append(1)
                last_event = i
            else:
                events.append(0)
        return pd.Series(events, index=sta_lta.index)

    def extract_features(self, data):
        """Extract features from seismic data."""
        sta_lta = self.calculate_sta_lta(data)

        # Dynamic threshold based on statistical properties
        dynamic_threshold = sta_lta.mean() + (2 * sta_lta.std())
        events = self.detect_seismic_events(sta_lta, dynamic_threshold)
        data['sta_lta'] = sta_lta
        data['event_detected'] = events

        # Diagnostic Logging
        logging.info(f"{self.metadata} STA/LTA - min: {sta_lta.min()}, max: {sta_lta.max()}, mean: {sta_lta.mean()}, std: {sta_lta.std()}")
        logging.info(f"{self.metadata} Dynamic Threshold: {dynamic_threshold}")

        return data, events, dynamic_threshold


class SampleGenerator:
    def __init__(self, window_size=500, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap

    def generate_negative_samples(self, data):
        """
        Generate negative samples by selecting windows with no detected events.
        """
        step = int(self.window_size * (1 - self.overlap))
        windows = []
        labels = []
        num_samples = len(data)

        for start in range(0, num_samples - self.window_size + 1, step):
            end = start + self.window_size
            window = data.iloc[start:end]
            if window['event_detected'].sum() == 0:
                # Extract aggregated features for the window
                agg_feat = {
                    'mean_normalized_amplitude': window['normalized_amplitude'].mean(),
                    'std_normalized_amplitude': window['normalized_amplitude'].std(),
                    'mean_sta_lta': window['sta_lta'].mean(),
                    'std_sta_lta': window['sta_lta'].std(),
                    'event_detected_count': window['event_detected'].sum()
                }
                windows.append(agg_feat)
                labels.append(0)  # Negative sample

        logging.info(f"Generated {len(windows)} negative samples.")
        return pd.DataFrame(windows), pd.Series(labels)

    def generate_positive_samples(self, data):
        """
        Generate positive samples by selecting windows with at least one detected event.
        """
        step = int(self.window_size * (1 - self.overlap))
        windows = []
        labels = []
        num_samples = len(data)

        for start in range(0, num_samples - self.window_size + 1, step):
            end = start + self.window_size
            window = data.iloc[start:end]
            if window['event_detected'].sum() > 0:
                # Extract aggregated features for the window
                agg_feat = {
                    'mean_normalized_amplitude': window['normalized_amplitude'].mean(),
                    'std_normalized_amplitude': window['normalized_amplitude'].std(),
                    'mean_sta_lta': window['sta_lta'].mean(),
                    'std_sta_lta': window['sta_lta'].std(),
                    'event_detected_count': window['event_detected'].sum()
                }
                windows.append(agg_feat)
                labels.append(1)  # Positive sample

        logging.info(f"Generated {len(windows)} positive samples.")
        return pd.DataFrame(windows), pd.Series(labels)


class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None

    def train(self, X, y):
        """Train a RandomForestClassifier on the extracted features and labels."""
        # Check if both classes are present
        classes = np.unique(y)
        if len(classes) < 2:
            logging.error("Insufficient classes for training. Ensure that both positive and negative samples are present.")
            sys.exit(1)  # Exit the program as training cannot proceed

        # Split the data into training and validation sets with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        logging.info(f"After SMOTE, counts of label '1': {sum(y_train_res == 1)}")
        logging.info(f"After SMOTE, counts of label '0': {sum(y_train_res == 0)}")

        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_res)
        X_val_scaled = self.scaler.transform(X_val)

        # Train the RandomForest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train_res)

        # Evaluate the model
        y_pred = self.model.predict(X_val_scaled)
        try:
            y_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        except AttributeError:
            logging.error("Model does not support probability predictions.")
            y_proba = np.zeros_like(y_pred)

        logging.info(f"Classification Report:\n{classification_report(y_val, y_pred)}")
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_val, y_pred)}")
        try:
            roc_auc = roc_auc_score(y_val, y_proba)
            logging.info(f"ROC AUC Score: {roc_auc}")
        except ValueError:
            logging.warning("ROC AUC Score cannot be calculated because only one class is present in y_true.")

        return self.model, self.scaler

    def save_model(self, model_path, scaler_path):
        """Save the trained model and scaler to disk."""
        import joblib
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logging.info(f"Model saved to {model_path}")
        logging.info(f"Scaler saved to {scaler_path}")


class Plotter:
    def __init__(self, planet, scaling_factor=1e26, max_points=100000):
        self.planet = planet
        self.scaling_factor = scaling_factor
        self.max_points = max_points

    def plot_seismic_data(self, data, events, dynamic_threshold, filename):
        """Plot the seismic data and detected events for a single CSV file using Plotly."""
        try:
            logging.info(f"Plotting data for file: {filename}")

            # Convert 'time' to datetime and set it as index
            data['time'] = pd.to_datetime(data['time'])
            data.set_index('time', inplace=True)

            # Log data shape
            logging.debug(f"Data shape for plotting: {data.shape}")

            # Scale normalized_amplitude for plotting
            data['normalized_amplitude_scaled'] = data['normalized_amplitude'] * self.scaling_factor

            # Log scaled normalized amplitude stats
            logging.debug(f"Scaled Normalized Amplitude Stats: min={data['normalized_amplitude_scaled'].min()}, max={data['normalized_amplitude_scaled'].max()}, mean={data['normalized_amplitude_scaled'].mean()}, std={data['normalized_amplitude_scaled'].std()}")

            # Subsample data if too large
            if len(data) > self.max_points:
                logging.warning(f"Data has {len(data)} points, which is more than the maximum allowed {self.max_points}. Subsampling.")
                data = data.sample(n=self.max_points, random_state=42).sort_index()

            # Create a figure with subplots
            fig = make_subplots(rows=3, cols=1,
                                shared_xaxes=True,
                                subplot_titles=(f'{self.planet} Seismic Data - Normalized Amplitude (Scaled)',
                                                f'{self.planet} Seismic Data - STA/LTA Ratio',
                                                f'{self.planet} Seismic Data - Detected Events'),
                                vertical_spacing=0.05)

            # Plot normalized amplitude (scaled)
            fig.add_trace(
                go.Scatter(x=data.index, y=data['normalized_amplitude_scaled'], mode='lines', name='Normalized Amplitude (Scaled)', line=dict(color='blue')),
                row=1, col=1
            )

            # Plot STA/LTA
            fig.add_trace(
                go.Scatter(x=data.index, y=data['sta_lta'], mode='lines', name='STA/LTA Ratio', line=dict(color='orange')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=[data['sta_lta'].mean()] * len(data), mode='lines', name='Mean STA/LTA', line=dict(color='green', dash='dash')),
                row=2, col=1
            )

            # Plot detected events
            event_times = events[events == 1].index
            event_values = [data['sta_lta'].max()] * len(event_times)
            fig.add_trace(
                go.Scatter(x=event_times, y=event_values, mode='markers', name='Detected Events', marker=dict(color='red', size=10)),
                row=3, col=1
            )

            # Update layout for better interactivity
            fig.update_layout(
                height=900,
                width=1200,
                title_text=f'{self.planet} Seismic Data Analysis - {filename}',
                showlegend=True
            )

            # Update x-axis to show date format with range slider and buttons
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1h", step="hour", stepmode="backward"),
                        dict(count=6, label="6h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                tickformat='%Y-%m-%d %H:%M',
                row=3, col=1
            )

            # Add y-axis titles
            fig.update_yaxes(title_text="Normalized Amplitude (Scaled)", row=1, col=1)
            fig.update_yaxes(title_text="STA/LTA Ratio", row=2, col=1)
            fig.update_yaxes(title_text="Detected Events", row=3, col=1)

            # Ensure the output directory exists
            html_dir = os.path.join("html_plots", self.planet)
            os.makedirs(html_dir, exist_ok=True)

            # Save the plot to an HTML file
            html_filename = os.path.join(html_dir, f"{self.planet}_seismic_data_plot_{os.path.splitext(os.path.basename(filename))[0]}.html")
            fig.write_html(html_filename)
            logging.info(f"Plot saved to {html_filename}")

        except Exception as e:
            logging.error(f"Error in plot_seismic_data_per_file for {filename}: {e}")


class SeismicAnalyzer:
    def __init__(self, planet):
        self.planet = planet
        self.planet_config = Config.get_planet_config(planet)
        if not self.planet_config:
            logging.error(f"Configuration for planet '{planet}' not found.")
            sys.exit(1)

        self.data_loader = DataLoader(self.planet_config)
        self.preprocessor = Preprocessor(self.planet_config)
        self.feature_extractor = FeatureExtractor(self.planet_config)
        self.sample_generator = SampleGenerator()
        self.model_trainer = ModelTrainer()
        self.plotter = Plotter(planet)

    def parse_arguments(self):
        """Parse command-line arguments to specify the planet."""
        parser = argparse.ArgumentParser(description='Seismic Event Detection for Mars or Lunar Data')
        parser.add_argument('planet', choices=['Mars', 'Lunar'], help='Planet to process data for')
        args = parser.parse_args()
        return args

    def load_training_data(self):
        """Load and preprocess training data."""
        training_data = []
        training_files = []
        for folder in self.planet_config['training_data_folders']:
            if not os.path.exists(folder['path']):
                logging.warning(f"Training folder does not exist: {folder['path']}")
                continue
            for filename in os.listdir(folder['path']):
                if filename.endswith('.csv'):
                    filepath = os.path.join(folder['path'], filename)
                    df, label = self.data_loader.load_data(
                        filepath,
                        folder['label']
                    )
                    if df is not None:
                        df_processed = self.preprocessor.preprocess_data(df)
                        df_features, events, dynamic_threshold = self.feature_extractor.extract_features(df_processed)
                        training_data.append((df_features, events, dynamic_threshold, filename))
                        training_files.append(filename)

        if not training_data:
            logging.error("No training data loaded. Exiting.")
            sys.exit(1)

        return training_data

    def process_and_collect_samples(self, training_data):
        """Process each training data file, plot, and collect samples."""
        all_features = []
        all_labels = []

        for df_features, events, dynamic_threshold, filename in training_data:
            # Plot seismic data per file with subsampling
            self.plotter.plot_seismic_data(df_features.copy(), events, dynamic_threshold, filename)

            # Generate positive and negative samples
            positive_X, positive_y = self.sample_generator.generate_positive_samples(df_features)
            negative_X, negative_y = self.sample_generator.generate_negative_samples(df_features)

            # Combine positive and negative samples
            X = pd.concat([positive_X, negative_X], ignore_index=True)
            y = pd.concat([positive_y, negative_y], ignore_index=True)

            all_features.append(X)
            all_labels.append(y)

        # Combine all features and labels from all files
        combined_X = pd.concat(all_features, ignore_index=True)
        combined_y = pd.concat(all_labels, ignore_index=True)

        logging.info(f"Total samples: {len(combined_y)}")
        logging.info(f"Positive samples: {(combined_y == 1).sum()}")
        logging.info(f"Negative samples: {(combined_y == 0).sum()}")

        # Ensure there are samples for both classes
        if combined_y.nunique() < 2:
            logging.error("Dataset does not contain both positive and negative samples. Exiting.")
            sys.exit(1)

        return combined_X, combined_y

    def run(self):
        """Execute the seismic analysis workflow."""
        # Load training data
        training_data = self.load_training_data()

        # Process data and collect samples
        combined_X, combined_y = self.process_and_collect_samples(training_data)

        # Train the model
        model, scaler = self.model_trainer.train(combined_X, combined_y)

        # Optionally, save the trained model and scaler for future use
        # model_filename = f"{self.planet}_random_forest_model.joblib"
        # scaler_filename = f"{self.planet}_scaler.joblib"
        # self.model_trainer.save_model(model_filename, scaler_filename)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Seismic Event Detection for Mars or Lunar Data')
    parser.add_argument('planet', choices=['Mars', 'Lunar'], help='Planet to process data for')
    args = parser.parse_args()

    # Initialize and run the seismic analyzer
    analyzer = SeismicAnalyzer(args.planet)
    analyzer.run()


if __name__ == "__main__":
    main()