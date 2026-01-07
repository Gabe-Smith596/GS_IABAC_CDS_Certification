# INX Future Inc. Employee Performance Prediction

A comprehensive machine learning project for predicting employee performance ratings using multiple classification algorithms and deep learning techniques.

## Project Overview

This project analyses employee data from INX Future Inc. to predict employee performance ratings (Good, Excellent, Outstanding) based on various employee attributes, work conditions, and satisfaction metrics. The analysis includes exploratory data analysis, feature engineering, machine learning and deep learning model training, and a web-based deployment interface. Additional insights into departmental trends are also analysed and business recommendations given.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Business Recommendations](#business-recommendations)
- [Technologies Used](#technologies-used)

## Dataset

**Source:** INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls

The dataset contains employee information including:
- Demographic data (Age, Gender, Marital Status)
- Job-related information (Department, Job Role, Job Level)
- Satisfaction metrics (Environment, Job Satisfaction, Work-Life Balance)
- Performance indicators (Hourly Rate, Salary Hike Percentage)
- Work experience metrics
- Performance ratings (Target variable)

**Class Distribution:**
- Most employees received "Excellent" ratings
- Notable class imbalance addressed through stratification and balanced class weights
- No employees with "Low" (1) performance ratings in the dataset

## Project Structure

```
Certification Project/
├── Gabriel Smith INX Performance Prediction Code Notebook.ipynb
├── README.md
├── Charts/
├── Model Deployment/
│   ├── employee_performance_app.py
│   ├── landing_page.py
│   ├── performance_prediction.py
│   ├── data_analysis.py
│   ├── requirements.txt
│   └── employee_performance/
└── Submission Directory Test/
```
## Methodology

### 1. Data Preprocessing
- Loaded and explored dataset structure
- Checked for missing values and duplicates (none found)
- Removed Employee Number column (unique identifier with no analytical value)
- Applied Label Encoding and One-Hot Encoding to categorical features

### 2. Exploratory Data Analysis
- **Distributions:** Analyzed performance ratings, age, commute distance, satisfaction metrics
- **Departmental Breakdowns:** Examined performance across departments
- **Relationships:** Investigated correlations between features
- **Principal Component Analysis:** Assessed feature separability

### 3. Machine Learning Models Evaluated
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier

### 4. Feature Engineering
- Conducted feature importance analysis using XGBoost
- Reduced feature set from 23 to 10 most important features

### 5. Deep Learning
- Built Sequential Neural Network with:
  - 5 Dense layers with ReLU activation
  - Batch Normalization layers
  - Dropout (30%) for regularization
  - Softmax output layer
- Applied early stopping to prevent overfitting
- Used balanced class weights

### 6. Model Selection
- Stratified K-Fold Cross Validation (5 folds)
- Evaluation metrics: Accuracy, Weighted F1 Score
- Classification reports for detailed per-class performance

## Key Findings

### Top 3 Features for Predicting Performance

1. **Employee Environment Satisfaction Rating** - Most important predictor
2. **Years Since Last Promotion** - Second most important
3. **Magnitude of Last Salary Hike** - Third most important

### Departmental Insights

- Large headcount disparities between largest and smallest departments
- Finance department showed the worst performance
- Environment Satisfaction significantly impacts performance across departments
- Work-Life Balance strongly correlates with performance ratings

### Notable Correlations

- Total Work Experience and Age: Moderate positive correlation
- Total Work Experience and Job Level: Strong positive correlation
- Years with Current Manager and Years in Current Role: Strong positive correlation (indicates slow promotion rate)

## Model Performance

### Best Performing Models

**Full Feature Set:**
- **XGBoost Classifier** achieved the best overall performance
- Used for feature importance analysis

**Reduced Feature Set (10 features):**
- **Random Forest Classifier** (default hyperparameters, balanced class weights)
- Selected for deployment
- Maintained strong balance of precision and recall across all classes
- Hyperparameter tuning confirmed default parameters were optimal

### Performance Metrics
- Stratified cross-validation ensured balanced class representation
- Weighted F1 Score prioritized to handle class imbalance
- Good precision and recall balance across all three performance rating classes

### Deep Learning Results
- Neural networks showed good performance on the majority "Excellent" class
- Lower performance on minority classes (likely due to dataset size)
- Neural networks typically require larger datasets for optimal performance

## Business Recommendations

### 1. Trial Flexible Working Arrangements
Work-Life Balance significantly impacts performance. Implement flexi-hours or work-from-home options where possible.

### 2. Data Collection on Low-Rated Employees
No "Low" performance ratings in dataset creates analytical blind spot. Collect comprehensive data across all performance levels.

### 3. Review Office Environment
As the top predictor of performance, invest in modernizing office spaces through updated decor, layout restructuring, or relocation.

### 4. Prioritize Internal Promotions
Time at company is a top-10 predictor. Favor internal promotions over external hires to leverage institutional knowledge.

### 5. Headcount Rebalancing in Finance
Finance department underperforming despite critical importance. Consider:
- Automation/AI augmentation in high-headcount departments (Sales, Development, R&D)
- Redirect resources to strengthen Finance team
- Implement AI tools for finance functions

## Technologies Used

### Data Science & ML
- **Python 3.x** - Primary programming language
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and utilities
- **XGBoost** - Gradient boosting framework

### Deep Learning
- **TensorFlow/Keras** - Neural network implementation

### Visualization
- **Matplotlib** - Base plotting library
- **Seaborn** - Statistical data visualization

### Deployment
- **Streamlit** - Web application framework
- **joblib** - Model serialization

### Development Tools
- **Jupyter Notebook** - Interactive development environment
- **VS Code** - Code editor

## Model Deployment

The trained Random Forest model has been exported along with:
- `employee_performance_rf_model.pkl` - Trained model
- `employee_performance_reduced_scaler.pkl` - Feature scaler
- `employee_department_encoder.pkl` - Department label encoder
- `employee_job_role_encoder.pkl` - Job role label encoder

These files enable real-time predictions through the Streamlit web application which can be found at the following address: https://gsiabaccdscertification-jlmddyo3yajkgdprehopjt.streamlit.app/<img width="1301" height="74" alt="image" src="https://github.com/user-attachments/assets/3d1f6389-61c9-4ce7-9b69-f0fa0347beb5" />


**Note:** This analysis addresses class imbalance through multiple techniques including stratified sampling, balanced class weights, and stratified K-fold cross-validation to ensure robust and fair model performance across all employee performance rating categories.
