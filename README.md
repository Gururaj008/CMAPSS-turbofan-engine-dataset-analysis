# CMAPSS-turbofan-engine-dataset-analysis

Turbofan Engine Remaining Useful Life (RUL) Prediction and Anomaly Detection

This repository contains three distinct machine learning projects focused on the Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset for turbofan engines. The primary goals are to predict the Remaining Useful Life (RUL) of engines and detect anomalies in their operational data.
Project Overview

The CMAPSS dataset provides time-series sensor data from turbofan engines under various operating conditions and fault modes. This repository explores different machine learning approaches to leverage this data for predictive maintenance.

    Random Forest Regressor for RUL Prediction: Implements a traditional machine learning approach using Random Forest to predict RUL, providing a robust and interpretable baseline.

    LSTM-based RUL Prediction for Turbofan Engines: Leverages Long Short-Term Memory (LSTM) networks to predict RUL, capturing temporal dependencies in the sensor data.

    Anomaly Detection using CORAL + Autoencoder: Focuses on identifying abnormal engine behavior using an unsupervised deep learning approach with domain adaptation.

1. 🌳 Random Forest Regressor for RUL Prediction

This project explores a traditional machine learning approach using a Random Forest Regressor to predict the Remaining Useful Life (RUL) of turbofan engines based on sensor data.
🔧 Input Data

    Dataset: Sensor data from turbofan engines from the CMAPSS dataset, specifically four training files (train_FD001.txt to train_FD004.txt), each representing different operating conditions.

    Content: Each engine unit has an engine ID, time in cycles, 3 operational settings, and 21 sensor measurements over time.

    Goal: Build a model that estimates how many cycles remain before engine failure based on these features.

🧹 Preprocessing

Several preprocessing steps were applied to prepare the data for the Random Forest model:

    Column Naming: Columns were renamed for clarity (e.g., sensor_1, op_setting_2).

    Data Loading & Merging: All four datasets were loaded, cleaned, and combined into a single DataFrame for unified processing.

    RUL Calculation: For each engine, RUL was computed as the difference between the maximum cycle count and the current cycle.

    Feature Selection: Less informative or constant-value columns were manually removed (e.g., 'op_setting_3', 'sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19').

    Normalization: Remaining features were normalized using MinMaxScaler to improve model performance.

    Feature Engineering (Rolling Window Features): Rolling mean and standard deviation features (with a window size of 20 cycles) were created for key sensors to capture degradation trends over time. This is particularly useful for time-series data.

📊 Train-Test Split

The preprocessed dataset was split into:

    Training Set: 80% of the data.

    Test Set: 20% of the data.

A fixed random state was used for reproducibility.
🤖 Model Architecture

A Random Forest Regressor was used with the following configuration:

    Number of Trees: 100.

    Random State: 42 (for reproducibility).

This non-parametric ensemble method builds multiple decision trees and averages their predictions, making it robust to overfitting and effective for tabular data.
📈 Results

After training, the model was evaluated on the test set using standard regression metrics. The Mean Absolute Error (MAE) was approximately 48.79, and the Root Mean Squared Error (RMSE) was around 64.98. A plot comparing actual and predicted RUL values for the first 100 samples in the test set showed that while not perfectly aligned, the predicted values generally follow the trend of the actual RUL, indicating reasonable model performance.
⚡ Significance and Implications

    Significance: This work demonstrates that a well-engineered Random Forest Regressor can provide useful predictions of engine degradation using only sensor and operational data, even without complex deep learning models. Key benefits include fast training and inference times, interpretability compared to black-box models, strong performance on tabular data, and serving as a good baseline for more advanced models.

    Implications: The model captures the general trend in RUL decay, suggesting that the selected features contain sufficient information about engine health progression. While it may struggle with extreme outliers or very long-term predictions, with further refinement (feature importance analysis, hyperparameter tuning, or stacking with other models), this approach could serve as a reliable part of a larger predictive maintenance system, especially where explainability is important.

2. 🚀 LSTM-based RUL Prediction for Turbofan Engines

This project focuses on predicting the Remaining Useful Life (RUL) of turbofan engines using time-series sensor data from the CMAPSS dataset, leveraging the power of Long Short-Term Memory (LSTM) networks.
🔧 Input Data

    Dataset: Time-series sensor data from the CMAPSS dataset. Four training files (train_FD001.txt to train_FD004.txt) are used, each representing different operating conditions and failure modes.

    Content: Each file contains engine unit ID, time in cycles, operational settings, and 21 sensor measurements over time.

    Goal: Build a sequence model that predicts how many cycles remain before engine failure based on historical sensor readings.

🧹 Preprocessing

Several preprocessing steps were applied to prepare the data for the LSTM model:

    Column Cleanup & Naming: Extra empty columns were dropped, and columns were renamed for clarity (e.g., sensor_measurement_1, op_setting_2).

    Dataset Combination: All four training datasets were concatenated into a single DataFrame for unified processing.

    RUL Calculation: For each engine unit, RUL was computed as the difference between the maximum cycle count and the current cycle, simulating "time to failure."

    Feature Selection: Less informative or constant-value columns were manually removed (e.g., 'op_setting_3', 'sensor_measurement_1', '5', '10', '16', '18', '19').

    Normalization: Remaining features were normalized using MinMaxScaler to improve model convergence.

🔄 Sequence Generation

To train the LSTM model, sequences of a fixed length (set to 50 cycles) were generated for each engine unit. Each sequence includes:

    A window of recent sensor readings and operational settings.

    The corresponding target RUL value for the last cycle in the window.

This approach allows the model to learn temporal patterns leading up to engine failure.
🤖 Model Architecture

A two-layer LSTM network was built using TensorFlow/Keras:

    Input Layer: Accepts sequences of shape (sequence_length, num_features).

    First LSTM Layer: 100 units with dropout (0.2).

    Second LSTM Layer: 50 units with dropout (0.2).

    Output Layer: Dense layer with 1 neuron (predicting scalar RUL).

    Loss Function: Mean Squared Error (MSE).

    Optimizer: Adam.

    Epochs: 20.

    Batch Size: 200.

    Training: The model was trained on generated sequences and validated using an 80/20 train-validation split.

📈 Results

The model was evaluated on the validation set, and predictions were compared against actual RUL values using a line plot. The plot shows a reasonable alignment between predicted and actual RUL trends, indicating that the model has learned meaningful degradation patterns from the sensor data.
⚡ Significance and Implications

    Significance: This work demonstrates a successful application of deep learning for predictive maintenance in aerospace systems. The model learns to predict remaining engine life using only sensor and operational data. The use of LSTMs effectively captures temporal dependencies, crucial for modeling progressive degradation.

    Implications: This approach can lead to early detection of performance degradation, reduced unplanned downtime through proactive maintenance scheduling, and potential integration into real-time monitoring systems. With further refinement (hyperparameter tuning, feature engineering, ensemble methods), this model could be a core component of a full-scale engine health monitoring system, capable of handling diverse operating conditions and fault types.

3. 📊 Anomaly Detection using CORAL + Autoencoder

This project aims to detect anomalies in turbofan engine test data by modeling normal behavior learned from training data using a deep autoencoder with a CORAL loss function for domain adaptation.
🔧 Input Data

    Dataset: Real-world sensor data from turbofan engines, specifically the CMAPSS dataset (train_FD001.txt and test_FD001.txt).

    Content: Each data point represents a time step for a given engine unit and includes multiple sensor readings and operational settings.

    Goal: Detect anomalies in test data by modeling normal behavior learned from training data.

🧹 Preprocessing

The following preprocessing steps were applied to ensure clean, consistent, and normalized input data:

    Column Removal: Irrelevant columns such as unit ID, time cycles, and three operational settings were removed.

    Empty Column Dropping: Empty columns were dropped to clean up the data.

    Normalization: Sensor features were normalized using MinMaxScaler to scale values between 0 and 1.

    Tensor Conversion: Final feature sets were converted into PyTorch tensors for compatibility with the deep learning framework.

🤖 Model Architecture

A fully connected Autoencoder was implemented to learn efficient representations of normal engine behavior.

    Encoder: Compresses the input into a lower-dimensional latent space.

    Decoder: Reconstructs the original input from the compressed form.

    Hybrid Loss Function: To improve generalization across train and test domains (especially when distributions differ slightly), the model was trained using a hybrid loss function combining:

        MSE Loss: Measures reconstruction accuracy.

        CORAL Loss (with weight 0.1): Aligns the covariance matrices of source and target domains to reduce domain shift.

    Training: The model was trained for 50 epochs using the Adam optimizer with a learning rate of 0.001.

📈 Results

    Reconstruction Errors: Calculated as the mean squared error between original and reconstructed inputs for the test set.

    Anomaly Threshold: Defined as the 95th percentile of all reconstruction errors (e.g., 0.0022).

    Anomaly Identification: Samples with reconstruction errors above this threshold were labeled as anomalies. Approximately 1–2% of the test samples were identified as anomalies.

⚡ Significance and Implications

    Significance: The model successfully detects subtle deviations in engine performance without relying on labeled failure data. Reconstruction error serves as a proxy for anomaly, identifying early signs of degradation. The CORAL loss improves adaptability to unseen test data, enhancing robustness for real-world deployment. Statistical validation confirms that detected anomalies represent structurally different patterns.

    Implications: This approach has direct applications in predictive maintenance systems for aerospace and industrial equipment. It enables early detection of abnormal behavior, helping to reduce unplanned downtime, optimize maintenance schedules, and improve overall system reliability.
