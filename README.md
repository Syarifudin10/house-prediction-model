# House Price Prediction Model 🏡📈

## Executive Summary:
Determining the fair market value of a property is a complex challenge involving numerous variables. Using Python and Scikit-Learn, I developed a machine learning pipeline to predict house prices based on the Boston Housing dataset. After performing rigorous data cleaning (including IQR outlier removal) and comparing three regression algorithms, the **Random Forest Regressor** emerged as the best performing model with an **R² Score of ~0.86**. The analysis highlights that the number of rooms and the neighborhood's socio-economic status are the primary drivers of real estate value.

### Business Problem:
Real estate investors and homeowners often struggle to estimate property values accurately due to market volatility and the subjective nature of manual appraisals. How can we leverage historical data to build an automated, data-driven valuation system that minimizes error?

The goal of this project is to build a predictive model that can ingest features (like crime rate, room count, etc.) and output a reliable price estimate.

![Model Comparison Result](https://github.com/Syarifudin10/house-prediction-model/blob/main/prediction_combined_model.png)
*(Figure 1: Comparison of Actual vs. Predicted values across three models. Note how Random Forest (Right) aligns most closely with the red diagonal line, indicating high accuracy.)*

### Methodology:
1.  **Data Preprocessing:**
    * Loaded the dataset and performed initial inspection for null values and duplicates.
    * Applied the **IQR (Interquartile Range) Method** to remove outliers in 'RM' (Rooms) and 'LSTAT' (Lower Status Population) to prevent skewed predictions.
    * Filtered out capped values (prices stuck at the maximum $50k) to ensure data integrity.

2.  **Exploratory Data Analysis (EDA):**
    * Visualized the distribution of the target variable (`MEDV`).
    * Used a **Correlation Heatmap** to identify which features strongly influence price.
    
    ![Correlation Matrix](https://github.com/Syarifudin10/house-prediction-model/blob/main/prediction_correlation.png)
    *(Figure 2: Heatmap showing strong positive correlation between Rooms (RM) and Price, and negative correlation between LSTAT and Price.)*

3.  **Model Development:**
    * Split the dataset into 80% training and 20% testing sets.
    * Trained and evaluated three distinct algorithms:
        * **Linear Regression:** Established a baseline.
        * **Decision Tree:** Tested for non-linear patterns.
        * **Random Forest:** Implemented an ensemble method to reduce overfitting and improve variance.

### Results & Findings:
* **Best Model:** The **Random Forest Regressor** achieved the highest accuracy (**R² ~0.86** and lowest MSE), proving to be the most stable model for this dataset.
* **Linear Regression** performed decently (**R² ~0.81**) but failed to capture complex non-linear relationships.
* **Decision Tree** showed signs of overfitting (**R² ~0.68**) with high variance in predictions.

**Key Insight:**
The analysis confirms that **'RM' (Number of Rooms)** is the strongest positive predictor of price, while **'LSTAT' (Lower Status Population)** is the strongest negative predictor.

### Next Steps:
1.  **Hyperparameter Tuning:** Utilize `GridSearchCV` to optimize `n_estimators` and `max_depth` for the Random Forest model.
2.  **Feature Engineering:** Explore interaction terms (e.g., Rooms per Capita) to see if they improve predictive power.
3.  **Deployment:** Save the model using `joblib` and deploy it as a simple web app using Streamlit for end-user interaction.
