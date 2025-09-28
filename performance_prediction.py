# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib


#Loading the Saved Components
model = joblib.load("employee_performance_rf_model.pkl")
scaler = joblib.load("employee_performance_reduced_scaler.pkl")
department_encoder = joblib.load("employee_department_encoder.pkl")
role_encoder = joblib.load("employee_job_role_encoder.pkl")


#Creating the Title Page
st.title("**Employee Performance Prediction**")
st.write("**Welcome to the Employee Performance Prediction page.**\n\n " \
"**Please use the Prediction section for generating employee performance predictions**.\n\n " \
"**Should you wish to find out more about the model development and performance metrics, please use the Model Development section.**")

# Creating the Performance Classes
classes = {"0": ["2", "good"], "1": ["3", "excellent"], "2": ["4", "outstanding"]}

# Creating the Departments and Job Roles
departments = ['Sales', 'Human Resources', 'Development', 'Data Science', 'Research & Development', 'Finance']
job_roles = ['Sales Executive', 'Manager', 'Developer', 'Sales Representative',
       'Human Resources', 'Senior Developer', 'Data Scientist',
       'Senior Manager R&D', 'Laboratory Technician',
       'Manufacturing Director', 'Research Scientist',
       'Healthcare Representative', 'Research Director', 'Manager R&D',
       'Finance Manager', 'Technical Architect', 'Business Analyst',
       'Technical Lead', 'Delivery Manager']

#Creating the Various DataFrames for the Model Development Section

initial_dict = {
    "Model Name": ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier", "XG Boost Classifier", "Five-Layer Artificial Neural Network"],
    "Accuracy": [0.695 , 0.876, 0.912, 0.927, 0.921, 0.827],
    "F1 Score": [0.714, 0.876, 0.908, 0.927, 0.920, 0.825],
    "Runtime (seconds)": [0.05 , 0.051, 1.257, 4.442, 2.806, 4.9]
}

initial_df = pd.DataFrame(initial_dict)

reduced_dict = {
    "Model Name": ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", "Gradient Boosting Classifier", "XG Boost Classifier", "Five-Layer Artificial Neural Network"],
    "Accuracy": [0.721, 0.885, 0.937, 0.931, 0.918, 0.816],
    "F1 Score": [0.739, 0.884, 0.935, 0.931, 0.917, 0.817],
    "Runtime (seconds)": [0.047, 0.032, 1.131, 2.792, 0.964, 8.4]
}

reduced_df = pd.DataFrame(reduced_dict)

classification_dict = {
    "Performance Rating Class": ["0 (Good)", "(1) Excellent", "(2) Outstanding"],
    "Random Forest Precision": ["0.86", "0.94", "1.00"],
    "Random Forest Recall": ["0.84", "0.97", "0.82"],
    "Random Forest f1 Score": ["0.85", "0.96", "0.90"]
}

classification_df = pd.DataFrame(classification_dict)

hyperparameter_dict = {
    "Parameters": ["Number of Estimators", "Max Depth", "Min Samples Split", "Min Samples Leaf", "Boostrap Sampling"],
    "Values Tested": [["50","100","150","200"], ["3","5","10","None"], ["2","5","10"], ["1","2","4"], ["True", "False"]]
}

hyperparameter_df = pd.DataFrame(hyperparameter_dict)

with st.expander("**Prediction**"):

    EMPDEP = department_encoder.transform([[st.selectbox("**Employee Department**", departments)]])[0]
    EMPROLE = role_encoder.transform([[st.selectbox("**Employee Role**", job_roles)]])[0]
    EMPSAT = st.slider("**Employee Environment Satisfaction Rating**", 1.0, 4.0, step = 1.0)
    EMPJOBINV = st.slider("**Employee Job Involvement Rating**", 1.0, 4.0, step = 1.0)
    SALHIKE = st.slider("**Last Salary Hike Percentage**", 10.0, 25.0, step = 1.0)
    WLB = st.slider("**Employee Work Life Balance Rating**", 1.0, 4.0, step = 1.0)
    YINX = st.slider("**Years of Experience at INX**", 0.0, 40.0, step = 1.0)
    CRE = st.slider("**Years of Experience in Current Role**", 0.0, 18.0, step = 1.0)
    PROMYEARS = st.slider("**Years Since Last Promotion**", 0.0, 15.0, step = 1.0)
    YCM = st.slider("**Years with Current Manager**", 0.0, 17.0, step = 1.0)

    #Preparing Input Features for Model
    features = np.array([[EMPDEP, EMPROLE, EMPSAT, EMPJOBINV, SALHIKE, WLB, YINX, CRE, PROMYEARS, YCM]])
    scaled_features = scaler.transform(features)

    #Employee Performance Rating Prediction
    if st.button("**Predict Employee Performance**", key="PredEmpPerformance"):
        
        prediction = model.predict(scaled_features)

        st.success(f"Predicted Employee Performance Rating: {classes[str(prediction[0])][0]}")
        if classes[str(prediction[0])][0] == "2":

            st.success(f"This indicates the employee is delivering a {classes[str(prediction[0])][1]} level of performance in their current role.")

        else:
            
            st.success(f"This indicates the employee is delivering an {classes[str(prediction[0])][1]} level of performance in their current role.")
        

with st.expander("**Model Development and Performance**"):

    st.write("""
            The model development process made use of a dataset comprised of 28 features with records for 1200 employees at INX. \n
            The aim was to develop a model that could predict employee performance rating with a high degree of accuracy based on the other gathered features. \n
            An imbalance in the number of samples between the different performance rating classes was discovered during the exploratory data analysis.
            This resulted in measures such the stratifying of the training and testing datasets by the number of samples in each class and the use of stratified k-fold cross validation during model evaluation.
            Models with the ability to set a class weights parameter were also set balanced class weights to prioritise equal performance between performance rating classes.
            The imbalance also prompted the testing of mainly tree-based models which are known to handle class imblances better than other model types. \n
            A 70/30 split was used for partitioning the dataset into training and testing datasets, this was done to promote better model generalisation. \n
            The table below gives a breakdown the model types tested as well as their initial performance using accuracy and weighted average f1 score during a stratified k-fold
            cross validation process using 5 folds and shuffling.
             """)
    
    st.dataframe(initial_df)

    st.write("""
            Taking the results together with the classification report outputs for each of the models into consideration, the XG Boost Classifier was deemed to be the best-performing model. \n
            It was used to generate a ranked list of feature importances towards predicting employee performance rating. 
            To balance performance and usability, only the top ten features were used to generate a reduced feature set for a secondary model evaluation. \n
            The table below gives a breakdown of the performance of the various model types on the reduced feature set.
    """)

    st.dataframe(reduced_df)

    st.write("""
            Based on both the cross validation and the classification reports, the Random Forest Classifier was deemed to be the best-performing model on the reduced feature set.
            The table below gives the output of the classification report.
    """)

    st.dataframe(classification_df)

    st.write("""
            Finally, hyperparameter tuning was conducted on the reduced feature set Random Forest Classifier to try improve it's performance further.
            This was done using Scikit Learn's GridSearchCV function.
            The table below gives a breakdown of the hyperparameters tuned and values trialled together with a random state of 42 and balanced class weights.
    """)

    st.dataframe(hyperparameter_df)

    st.write("""
            The default hyperparameter initialisaion values were included in the list trialled and were returned as the best-performing configuration. 
            This configuration was thus used in the model exported for this deployment.
    """)
