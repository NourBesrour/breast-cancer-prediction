# Breast Cancer Diagnosis Prediction App

This project is a web-based application built using Streamlit and machine learning to predict whether a breast tumor is benign or malignant based on several medical features. The machine learning model is trained using a breast cancer dataset and utilizes a logistic regression algorithm to make predictions.

## Features
- **Interactive interface**: Users can input specific medical details related to a patient's breast cancer condition, and the app will predict whether the tumor is malignant or benign.
- **Machine Learning**: A logistic regression model has been trained on historical breast cancer data and integrated into the app for prediction.
- **Simple and user-friendly design**: The app is built using Streamlit, making it easy to use and visually intuitive.

## Dataset
The dataset used in this project contains information related to breast cancer patients, including:
- Age
- Tumor size (in cm)
- Number of involved lymph nodes
- Whether the patient is pre- or post-menopausal
- Breast quadrant affected
- Family history of cancer

The dataset does not contain missing values, and some irrelevant columns (e.g., serial number, year) have been removed for model training purposes.

## Model
The machine learning model used in this project is Logistic Regression with l1 regularization and the liblinear solver.

**Model Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score

These metrics help evaluate the performance of the model on both the training and test datasets.

## Requirements
- Python 3.7+
- Streamlit 1.15.2+
- Scikit-learn 1.0.2+
- Pandas 1.4.3+
- Matplotlib 3.5.2+
- Seaborn 0.11.2+


## Files in the Repository
- `app.py`: The Python script that trains the Logistic Regression model and performs the predictions.
- `main.py`: Contains the main logic to execute the logistic regression model.
- `breast-cancer-dataset.csv`: The dataset used to train the model.
- `BC.png`: An image used in the Streamlit app to represent breast cancer.
- `xgbpipe.joblib`: The saved pipeline or model used for predictions in the Streamlit app.
- `README.md`: This file explaining the project.
- `requirements.txt`: List of Python libraries and their versions required to run the app.

## How to Run the App
1. Clone the repository:
    ```bash
    git clone https://github.com/NourBesrour/breast-cancer-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd breast-cancer-prediction
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

The app should launch in your default web browser. You can then input the patient details and get a prediction on whether the tumor is benign or malignant.

## Usage
Once the app is running, follow these steps to make a prediction:

1. Select values for each of the following:
    - Menopause status
    - Involved nodes (axillary lymph nodes)
    - Breast side (left or right)
    - Whether metastasis is present
    - Breast quadrant
    - History of cancer
2. Enter the age of the patient at diagnosis and the tumor size.
3. Click **Predict**. The app will display either "Malignant" or "Benign" based on the model's prediction.

## Model Evaluation
The model was evaluated on various metrics:
- **Accuracy**: How often the model correctly classifies tumors.
- **Precision**: How many of the predicted malignant tumors are actually malignant.
- **Recall**: How many of the actual malignant tumors were correctly classified.
- **F1 Score**: A balance between precision and recall.

## Future Improvements
Some potential improvements for this project include:
- Adding more advanced models (e.g., XGBoost, Random Forest) for comparison with logistic regression.
- Implementing hyperparameter tuning for better model performance.
- Collecting additional data for more accurate predictions.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
