# LSTM for Battery State-of-Health Prediction

This project implements a Long Short-Term Memory (LSTM) neural network to predict the State-of-Health (SOH) and Remaining Useful Life (RUL) of Lithium-ion batteries, using the NASA Prognostics Battery Dataset.

The primary goal is to predict the battery's capacity degradation over its lifecycle by learning from raw time-series sensor data.

## Dataset

This model uses the [NASA Prognostics Center of Excellence (PCoE) Battery Data Set](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). The dataset contains measurements from various Li-ion batteries run under different operational profiles (charge, discharge, impedance) until they reached their end-of-life (EOL), defined as a 30% fade in rated capacity.

Our model focuses on the `discharge` cycles, using the following time-series data as input:
*   Voltage Measured
*   Current Measured
*   Temperature Measured

The target variable is the battery's `Capacity` in Ampere-hours (Ah).

## Methodology

The core of this project is a Many-to-One LSTM architecture.

1.  **Data Preprocessing**:
    *   Each battery's complete lifecycle data is loaded from its `.mat` file.
    *   Discharge cycles are isolated.
    *   Since each cycle has a different duration, all time-series data is interpolated to a fixed length of 500 time steps.
    *   The features (Voltage, Current, Temperature) are stacked into a 3D tensor of shape `(num_cycles, 500, 3)`.

2.  **Data Splitting (By Battery ID)**:
    *   To ensure the model is tested on its ability to generalize to new, unseen batteries, the data is split by battery ID. This avoids the critical pitfall of chronological data leakage.
    *   **Training Set**: Full lifecycle data from batteries `B0005` and `B0006`.
    *   **Validation Set**: Full lifecycle data from battery `B0007`.
    *   **Test Set**: Full lifecycle data from battery `B0018`.

3.  **Model Architecture**:
    *   The model consists of an LSTM layer followed by a Dropout layer for regularization and a final Dense (Linear) layer to output the predicted capacity.
    *   It processes the entire sequence of 500 time steps and uses the final hidden state for prediction.

## Usage

1.  Download the NASA battery dataset and place the `.mat` files in the `data/` directory.
2.  Update the `data_directory` path and the battery ID lists in `main.py` to match your setup.
3.  Run the main script to start the training and evaluation process:
    ```bash
    python main.py
    ```
    The script will output training progress, final test metrics, and save plots of the loss history and test predictions.

## Current Results

The model successfully learns the degradation pattern on the training data, achieving a near-perfect R² score. However, it exhibits a classic case of **overfitting**, where its performance on the unseen validation and test batteries is significantly lower.

*   **Training R²:** ~0.99
*   **Test R²:** ~0.48
*   **Test RMSE:** ~0.126

This performance gap indicates that while the LSTM is powerful enough to memorize the patterns of the training batteries, it struggles to generalize to the unique degradation signatures of new batteries. The prediction plot shows the model correctly captures the downward trend but has a systematic bias, consistently over-predicting the capacity.

---

## Future Work: Improving Generalization with Feature Engineering

The current model's primary limitation is its difficulty in generalizing from raw sensor data alone. The most effective way to improve performance is through **feature engineering**, which involves creating new, highly informative input features for the model.

### Learning from the Kaggle Notebook's Approach

A highly successful public Kaggle notebook on this dataset achieves an R² score of **~0.98** on the test set. It does this not with a complex model, but by solving a much simpler, cleverly framed problem.

Instead of feeding raw sensor curves to a model, the notebook **engineers a few powerful features** and uses a simpler Random Forest model. Here’s exactly what it does:

1.  **It Ignores the Raw Curves:** The notebook's final model does not use the voltage, current, or temperature sequences at all. It relies entirely on summary features.

2.  **It Creates a Normalized Target (SOH):** It predicts the State-of-Health (SOH) instead of absolute capacity.
    *   `SOH = Measured_Capacity / Rated_Capacity`
    *   This normalizes the target to a 0-1 scale and makes the problem more consistent across different batteries.

3.  **It Uses the Cycle Number as a Key Feature:** The most direct indicator of a battery's age is its cycle number. This is a powerful "shortcut" feature that immediately tells the model where it is in the degradation process.

4.  **It Uses Lag Features (Historical SOH):** To make a prediction for the current cycle, it uses the SOH from the *previous one or two cycles*. This transforms the problem from "predict from physics" to "predict the next step in a sequence." The model is asked: "Knowing the health was 95% yesterday and 94.5% the day before, what is it today?" This is a much easier question to answer.

5.  **It Uses Logarithmic Transformation:** The notebook applies a `log()` function to the `Cycle` number and `SOH` values. This mathematically transforms the curved degradation trend into a nearly straight line, making the pattern trivial for a simple model to learn.

### How to Implement This in Your Project

You can dramatically improve your model by adopting a hybrid approach:

1.  **Engineer New Features:** In `data_preprocess.py`, create new features for each cycle:
    *   `SOH`
    *   `Cycle_Number`
    *   Lag features like `SOH_previous_cycle`

2.  **Create a Hybrid Model:** Your LSTM is excellent at learning from sequences. Don't discard it—use it as a powerful feature extractor.
    *   **Step A (LSTM Feature Extractor):** Keep your LSTM, but have it process the raw curves and output a small, dense vector (e.g., of size 8 or 16). This vector becomes a learned "health summary" of the raw physical signals for that cycle.
    *   **Step B (Combine Features):** Concatenate the LSTM's output vector with the handcrafted features (`Cycle_Number`, `SOH_previous_cycle`, etc.).
    *   **Step C (Final Prediction):** Add a small feed-forward network (a few Dense layers) after the concatenation step to make the final SOH prediction from this rich, combined feature set.

By combining the deep learning power of your LSTM to interpret raw data with the focused, problem-simplifying power of handcrafted features, you can build a far more accurate and robust model.
