
# Heart Disease Prediction ML Project

This project predicts heart disease outcomes using multiple machine learning models implemented in a single Jupyter Notebook. The workflow includes data preprocessing, exploratory data analysis, feature selection, feature scaling, model training, and performance evaluation.

## Project Structure

- `heart_disease_prediction_project.ipynb` - Complete machine learning workflow, including data preprocessing and experiments with multiple classification models.
- `Dataset/heart disease prediction dataset.csv` - Heart disease dataset used by the notebook.

## Dataset

The project uses a CSV dataset containing 1,025 records and 14 columns, including clinical and demographic features related to heart disease.

The dataset includes the following features:

- `age` - Age of the individual.
- `sex` - Sex of the individual, encoded numerically.
- `cp` - Chest pain type.
- `trestbps` - Resting blood pressure.
- `chol` - Serum cholesterol level.
- `fbs` - Fasting blood sugar indicator.
- `restecg` - Resting electrocardiographic results.
- `thalach` - Maximum heart rate achieved.
- `exang` - Exercise-induced angina indicator.
- `oldpeak` - ST depression induced by exercise relative to rest.
- `slope` - Slope of the peak exercise ST segment.
- `ca` - Number of major vessels, as represented in the dataset.
- `thal` - Thalassemia-related categorical value.
- `target` - Target variable used for heart disease classification.

### Target Labels

The `target` column contains two classes:

- `0` = No heart disease.
- `1` = Heart disease present.

The dataset contains 499 records labeled `0` and 526 records labeled `1`.

## Workflow

The notebook follows this general pipeline:

1. Load the heart disease dataset.
2. Inspect the dataset's structure and information.
3. Remove the `sex` column.
4. Convert selected columns to numeric data types.
5. Identify and remove duplicate rows.
6. Analyze the distributions and skewness of continuous features.
7. Apply the Box-Cox transformation to selected continuous features.
8. Detect and cap outliers using the Interquartile Range (IQR) method.
9. Handle missing values using mean imputation.
10. Separate the features and target variable.
11. Analyze feature correlations and remove highly correlated features using a threshold of 0.90.
12. Split the dataset into training and testing sets using an 80:20 ratio.
13. Standardize selected continuous features using `StandardScaler`.
14. Train multiple machine learning classification models.
15. Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.
16. Perform 10-fold cross-validation to assess model performance.
17. Compare training and testing accuracy to check for potential overfitting.

## Machine Learning Models

The notebook implements the following classification algorithms:

- **Logistic Regression** - Estimates the probability of a sample belonging to a particular class.
- **Random Forest** - Combines multiple decision trees to make predictions.
- **K-Nearest Neighbors (KNN)** - Classifies samples based on their nearest neighbors.
- **Gaussian Naive Bayes** - Uses Bayes' theorem with a Gaussian distribution assumption for continuous features.

Each model is trained and evaluated using the processed dataset.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures how many positive predictions are correct.
- **Recall:** Measures how many actual positive cases are correctly identified.
- **F1-score:** Balances precision and recall.
- **Confusion Matrix:** Summarizes correct and incorrect predictions for each class.
- **10-fold Cross-Validation:** Evaluates model performance across multiple data splits.
- **Overfitting Check:** Compares training and testing accuracy to identify potential overfitting.

## Requirements

The project uses the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`

## How To Run

1. Download or clone the project repository.
2. Ensure the `Dataset/` folder contains `heart disease prediction dataset.csv`.
3. Open `heart_disease_prediction_project.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.
4. Update the dataset path in the notebook if necessary. For example, in Google Colab:

   ```python
   BASE_PATH = "/content/Dataset/heart disease prediction dataset.csv"
   ```

5. Run the notebook cells from top to bottom.
6. Review the evaluation metrics to compare the performance of the different models.

## Notes

- All four machine learning models are implemented in the same notebook.
- The notebook expects the dataset to contain a `target` column.
- This project is intended for educational and research purposes. Its predictions should not be used as a substitute for professional medical evaluation.
