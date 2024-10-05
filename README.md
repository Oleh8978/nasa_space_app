Seismic Event Detection

This project implements a comprehensive seismic event detection system specifically designed for analyzing seismic data from Mars and the Moon. The program utilizes various libraries, including pandas, scikit-learn, and plotly, to preprocess data, extract features, train a machine learning model, and visualize results.

Table of Contents
1. Installation
2. Usage
3. Components
4. Configuration
5 .Logging
6. License

Installation:
  To set up this project, you'll need to install the necessary dependencies. You can do this using pip:
  pip install pandas numpy scikit-learn imbalanced-learn plotly joblib

Usage:
  Run the script via command line, providing the target planet as an argument:
  python seismic_event_detection.py [planet]
  Replace [planet] with either Mars or Lunar.
  python seismic_event_detection.py Mars

Components
1. Config
Handles loading configurations from a JSON file. The configuration includes parameters such as gravity, mass, and header mappings specific to the planet.

2. DataLoader
Loads seismic data from CSV files, checks for necessary columns, renames them, converts data types, and performs preliminary logging.

3. Preprocessor
Normalizes the amplitude of the seismic data based on the planet's gravity and mass.

4. FeatureExtractor
Calculates features from the data, including:

Short-Term Average to Long-Term Average (STA/LTA) ratios.
Event detection based on dynamic thresholds.
5. SampleGenerator
Generates positive and negative samples from the data for training purposes. This ensures balanced datasets for the machine learning model.

6. ModelTrainer
Trains a Random Forest Classifier on the extracted features, handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique), and evaluates the model performance using metrics such as ROC AUC score.

7. Plotter
Creates interactive visualizations of the seismic data, STA/LTA ratios, and detected events using Plotly.

8. SeismicAnalyzer
Orchestrates the entire workflow from loading data, processing, sample generation, training the model, and plotting results.

Configuration
The configuration for the planets is stored in a JSON file (e.g., config.json). The file should have the following structure:

{
    "Mars": {
        "gravity": 3.721,
        "mass": 6.4171e23,
        "threshold": 2.0,
        "metadata": "02.BHV",
        "header_mapping": {
            "time_column": "time(%Y-%m-%dT%H:%M:%S.%f)",
            "rel_time_column": "rel_time(sec)",
            "velocity_column": "velocity(c/s)"
        },
        "velocity_unit_conversion": 1.0,
        "training_data_folders": [
            {
                "path": "./data/mars/training/data/",
                "label": 1
            }
        ],
        "test_data_folders": [
            {
                "path": "./data/mars/test/data/",
                "label": 1
            }
        ]
    },
    "Lunar": {
        "gravity": 1.62,
        "mass": 7.342e22,
        "threshold": 2.0,
        "metadata": "12.00.mhz",
        "header_mapping": {
            "time_column": "time_abs(%Y-%m-%dT%H:%M:%S.%f)",
            "rel_time_column": "time_rel(sec)",
            "velocity_column": "velocity(m/s)"
        },
        "velocity_unit_conversion": 100.0,
        "training_data_folders": [
            {
                "path": "./data/lunar/training/data/S12_GradeA/",
                "label": 1
            }
        ],
        "test_data_folders": [
            {
                "path": "./data/lunar/test/data/S12_GradeB/",
                "label": 0
            },
            {
                "path": "./data/lunar/test/data/S15_GradeA/",
                "label": 1
            },
            {
                "path": "./data/lunar/test/data/S15_GradeB/",
                "label": 0
            },
            {
                "path": "./data/lunar/test/data/S16_GradeA/",
                "label": 1
            },
            {
                "path": "./data/lunar/test/data/S16_GradeB/",
                "label": 0
            }
        ]
    }
}

Logging
The application utilizes Python's built-in logging module to log information, warnings, and errors throughout the execution. Logs provide insights into the data loading, preprocessing, model training, and plotting stages, assisting with debugging and monitoring.

License
This project is licensed under the MIT License. See the LICENSE file for details.



