![Data Loading](./Data/images/lift-top-banner.png)


# Lyft Price Prediction

### Author: Ajay Varma  
### Date: April 2025  
### Project Directory: `/Users/ajayvarma/Documents/VS Code/Workspace/Data Science/Projects/Lyft/`  
### Type: Portfolio Project  
---

## Project Overview

### Objective
The objective of this project is to build a machine learning model that predicts the price of Lyft rides. The model takes into account various features such as ride distance, passenger count, time of day, and ride type. By providing accurate pricing predictions, this model helps improve pricing transparency, customer satisfaction, and operational efficiency for Lyft.

## Lyft VSCode Setup

![Lyft VSCode 1](https://github.com/ajayvarmaco/Lyft/blob/main/Data/images/lyft-vscode-1.png)

![Lyft VSCode 2](https://github.com/ajayvarmaco/Lyft/blob/main/Data/images/lyft-vscode-2.png) ![Lyft VSCode 3](https://github.com/ajayvarmaco/Lyft/blob/main/Data/images/lyft-vscode-3.png)

### Dataset
The dataset consists of historical Lyft ride data and includes the following key features:

- **`distance`**: Ride distance in miles.
- **`passenger_count`**: Number of passengers in the ride.
- **`time_of_day`**: Time of day when the ride was requested.
- **`price`**: Ride price (target variable).
- **Categorical Features**: Ride type (`name_Lux Black XL`, `name_Shared`), weather conditions, and source/destination locations (`source_North End`, `destination_Back Bay`).

### Methodology
The project follows a structured approach to machine learning model development:

1. **Data Preprocessing**:
   - Cleaned the data by removing duplicates, handling missing values, and encoding categorical variables using **one-hot encoding**.
   
2. **Outlier Treatment**:
   - Applied the **Interquartile Range (IQR)** method to detect and remove outliers from the `price` and `distance` columns, ensuring better model accuracy.

3. **Feature Engineering**:
   - Selected and directly used features such as `distance`, `passenger_count`, and `time_of_day` for model training. Categorical features like ride types and weather conditions were encoded for use in machine learning models.

4. **Model Development**:
   - Built and compared multiple regression models: **Linear Regression** and **Random Forest Regressor**. The **Random Forest** model outperformed **Linear Regression** due to its ability to capture non-linear relationships and complex feature interactions.

5. **Model Evaluation**:
   - Evaluated models using **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R-squared (R²)** to measure accuracy and predictive power.

6. **Feature Importance**:
   - Conducted feature importance analysis using the **Random Forest** model to identify the most influential features. The ride type (`name_Lux Black XL`) and distance were found to be the most important features for predicting price.

7. **Model Saving**:
   - The best-performing model, **Random Forest**, was saved using **joblib** for future use and deployment.

---

## Model Performance Comparison

| Model            | MAE (Mean Absolute Error) | MSE (Mean Squared Error) | R-squared |
|------------------|----------------------------|--------------------------|-----------|
| **Linear Regression** | 1.63                       | 4.52                     | 0.95      |
| **Random Forest**     | 0.95                       | 2.07                     | 0.98      |

---

## Insights & Recommendations

### Key Insights
1. **Random Forest Outperforms Linear Regression**:
   - **Lower MAE** (0.95 vs. 1.63): **Random Forest** delivers predictions that are much closer to actual values.
   - **Lower MSE** (2.07 vs. 4.52): The **Random Forest** model exhibits smaller squared errors on average, improving the prediction quality.
   - **Higher R²** (0.98 vs. 0.95): **Random Forest** explains 98% of the variance in the data, providing a more accurate prediction of the ride prices.

2. **Feature Importance**:
   - The **most important features** for price prediction were:
     - **Ride Type** (`name_Lux Black XL`)
     - **Distance**
   - **Weather conditions** and **source/destination features** were less influential in the model’s predictions.

### Recommendations
1. **Model Deployment**:
   - The **Random Forest** model is recommended for deployment due to its superior performance. It can be used for real-time ride price prediction in Lyft's production environment.

2. **Model Tuning**:
   - **Hyperparameter tuning**: Consider adjusting parameters such as the number of trees and tree depth in the **Random Forest** to improve model performance.
   - **Cross-validation**: Implement **cross-validation** techniques to ensure the model generalizes well across different data subsets.

3. **Exploring Advanced Models**:
   - Experiment with advanced models like **Gradient Boosting Machines (GBM)**, **XGBoost**, or **LightGBM** to explore further performance improvements.

4. **Feature Engineering**:
   - Consider adding new features or transforming existing ones to further enhance model accuracy.
   - **Feature selection** could be useful to remove irrelevant or highly correlated features, improving the model's efficiency.

5. **Outlier Handling**:
   - Although **IQR outlier removal** was effective, additional domain knowledge or custom outlier detection techniques could be applied to refine the model.

6. **Future Work**:
   - **Model Monitoring and Maintenance**: Monitor the model's performance over time and retrain it as new data becomes available to maintain prediction accuracy.
   - **Scalability**: Ensure the model is scalable to handle larger datasets as Lyft expands.

---

## Conclusion
- The **Random Forest** model is the recommended choice for predicting Lyft ride prices due to its exceptional performance across key metrics (MAE, MSE, R²).
- While **Linear Regression** provides a simpler alternative, **Random Forest** outperforms it in capturing complex relationships in the data.
- **Hyperparameter tuning**, **exploring new models**, and **continuous monitoring** are recommended for ongoing improvements in the model's predictive performance.

---

## Project Structure

- `Notebooks/`: Contains Jupyter notebooks used for exploratory data analysis, data preprocessing, and model training.
- `Models/`: Contains the trained machine learning models, including the saved **Random Forest model** (`random_forest_model.joblib`).
- `Data/`: Contains datasets and other related files, such as data used for training and testing.


![Data Loading](./Data/images/lift-bottom.png)
