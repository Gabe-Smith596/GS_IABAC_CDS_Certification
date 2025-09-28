# Importing Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image


#Creating the Title Page
st.title("**INX Employee Performance Study Data Analysis**")
st.write("""
        Welcome to the Data Analysis page for the INX Employee Performance Study. 
        On this page you will find the charts and graphics generated during the exploratory data analysis as well as the feature importance analysis conducted during the model fitting and the final business recommendations given.
        There are three sections of charts and a section each for feature importances and business recommendations, each contained within a collapsable header.
    """)

with st.expander("**Distributions**"):

    emp_peformance_distibution = Image.open("Employee Performance Rating Distribution.png")
    st.image(emp_peformance_distibution, caption="Most employees received a performance rating of Excellent. Notably there were no employees that received a Low (1) performance rating. There is quite a pronounced imbalance in the number of employees between the three classes which needed to be factored into the training of models in future to avoid sub-optimal model performance on Good and Outstanding ratings.")

    emp_age_distribution = Image.open("Employee Age Distribution.png")
    st.image(emp_age_distribution, caption="The company had the a wide distribution of employee ages from 18 years to 60 with the most common employee cohort being the 32-34 years age range. ")

    emp_commute_distance = Image.open("Employee Commute Distance Distribution.png")
    st.image(emp_commute_distance, caption="All employees commuted fewer than 30 kilometres* to get to the office. Most employees only needed to commute 0-2.5 kilometres.*An assumption was made that the data recorded were in kilometres given that no location was specified for the company and the kilometre is one of the SI units for distance.")

    emp_environment_satisfaction_distribution = Image.open("Employee Environment Satisfaction Ratings Distribution.png")
    st.image(emp_environment_satisfaction_distribution, caption="The majority of employees had favourable view (high or very high rating) of the office environment.")

    emp_job_satisfaction_distribution = Image.open("Employee Job Satisfaction Distribution.png")
    st.image(emp_job_satisfaction_distribution, caption="The majority of employees had a favourable view (high or very high rating) of their satisfaction with their job.")

    emp_hourly_earnings_distribution = Image.open("Employee Hourly Rate Distribution.png")
    st.image(emp_hourly_earnings_distribution, caption="Distribution appeared to be largely uniform with the largest cohort of employees earning between $40-45* *Similarly to the commute distance distribution, due to no geographical location being specified for the company, an assumption was made that the hourly rate data were priced in US dollars given that it is the world's reserve currency and forms the basis of most of the world's trade.")

    emp_most_recent_salary_hike_distribution = Image.open("Employee Most Recent Salary Hike Distribution.png")
    st.image(emp_most_recent_salary_hike_distribution, caption="The pay increases appeared to be heavily right-skewed with certain employees able to command outsized salary advancements. The largest cohort of employees received pay increases of 11%.")

with st.expander("**Departmental Breakdowns**"):

    num_emp_per_department = Image.open("Heatcount per Department.png")
    st.image(num_emp_per_department, caption="There were large differences in the employee headcount between the three largest departments and the three smallest.")

    performance_per_department = Image.open("Average Employee Performance Rating per Department.png")
    st.image(performance_per_department, caption="Only two departments were able to achieve an overall average performance rating of Excellent (3) with rest being rated in the Good category(2-3). Worryingly, the Finance department was the worst-performing department.")

    performance_per_department_env = Image.open("Average Employee Performance Rating per Department by Environment Satisfaction Rating.png")
    st.image(performance_per_department_env, caption="Across five of the six departments, those with Low (1) or Medium (2) environment satisfaction ratings had lower average performance ratings than those with more favourable views on the office environment. Given the magnitude of the difference between the lower and upper scale rating classes, this was identified as a potentially significant differentiating factor for employee performance.")

    performance_per_department_work_life = Image.open("Average Employee Performance Rating per Department by Work Life Balance Rating.png")
    st.image(performance_per_department_work_life, caption="Similarly to Environment Satisfaction, across five of the six departments, those with Low (1) or Medium (2) work life balance ratings had lower average performance ratings than those with more favourable views. In the Finance department, even those with highly-rated work life balance are achieving sub-optimal performance ratings. Given the low headcount, this could indicate a work overload.")

    job_satisfaction_by_overtime = Image.open("Employee Job Satisfaction per Department by Overtime Requirement.png")
    st.image(job_satisfaction_by_overtime, caption="The breakdown of job satisfaction per department by overtime requirement yielded some interesting results. In certain departments such as Sales, Data Science, Human Resources and Finance employees who were required to work overtime had higher average job satisfaction ratings than those that were not required. However, the Research & Development and Development departments showed adverse job satisfaction ratings for employees required to work overtime.")

    num_overtime_per_dep = Image.open("Number of Employees per Department by Ovetime Requirement.png")
    st.image(num_overtime_per_dep, caption="All departments had a higher number of employees not required to work overtime than those required to. The Development department had the highest proportion of overtime-required employees at 49.79%")

    work_life_per_job_level = Image.open("Employee Work Life Balance Ratings per Department by Job Level.png")
    st.image(work_life_per_job_level, caption="The breakdown of employee work life balance per department and then by job level did not reveal any universal trends such as an increase or decrease in work life balance with increasing for decreasing job level. However, two noticeable findings were very poor work life balance ratings ('Bad'. 1) for the second-highest (4) job level in the Data Science department as well as lower average rating for the highest job level (5) in the Finance department.")


with st.expander("**Correlations**"):

    pca = Image.open("Employee Performance Ratings Principal Component Analysis.png")
    st.image(pca, caption="There did not appear to be a clear separation betweeen the three performance rating classes.")

    heatmap = Image.open("Employee Features Correlation Heatmap.png")
    st.image(heatmap)

    st.markdown("""
    **Notable Correlations**
 
    - Total Years of Work Experience and Age are moderately positive correlated.
    - Total Years of Work Experience and Job Level are strongly positively correlated.
    - Business Travel Frequency categories are moderately negatively correlated.
    - Marital Status categories are moderately negatively correlated.
    - Years of Experience at the Company and Years with Current Manager are strongly positively correlated. This combined with the strong positive correlation with Years of Experience in Current Role indicate a slow rate of progression in the job level strucuture within the company through promotion. 

    **Notable Correlations with Employee Performance Rating**

    - Employee Environment Satisfaction and the magnitude of the last salary increase in percentage terms were the two most highly correlated features to Employee Performance Rating with moderate positive correlations of 0.4 and 0.33, respectively. 

    **Multicollinearity**

    - Using a threshold of 0.8, there were no instances of multicollinearity detected based on the above feature correlations heatmap.
    """)

with st.expander("**Feature Importance Analysis**"):

    feature_imps = Image.open("Feature Importances.png")
    st.image(feature_imps, caption="Employee Environment Satisfaction was ranked as the most important feature for predicting employee performance. It is unclear whether this is referring to the actual decor/infrastructure of the office or the collection of relationships in the office. For the purpose of this analysis, the assumption will be made that it was referring to the office decor/aesthetic. Three measures in the top ten refer directly or indirectly to time spent in a particular role: Years Since Last Promotion, Experience Years in Current Role and Years with Current Manager. The magnitude of the last salary hike was also notably high in predictive power for employee performance.")

with st.expander("**Business Recommendations**"):

    with st.expander("**Best-Performing Model**"):
        st.markdown("""
            **Best Performing Model**

            - The XG Boost Classifier was the best-performing model on the full feature set. This model was used for the feature importance analysis which resulted in the reduced feature set to improve usability.
            - The Random Forest Classifier with default intialisation hyperparameters, a random state of 42 and balanced class weights was the best-performing model on the reduced feature set and was exported for for a web-based model deployment.
                    """)
    
    with st.expander("**Feature Importance for Predicting Employee Performance**"):

        st.markdown("""
            **Feature Importance for Predicting Employee Performance**

            The top three most important features for predicting employee performance were:
            - Employee Environment Satisfaction Rating
            - Years Since Last Promotion
            - Magnitude of the Last Salary Hike
                    """)
        
    with st.expander("**Departmental Trends**"):

        st.markdown("""
            **Departmental Trends**

            - Large headcount differences between the three largest and three smallest departments.
            - Finance department the worst-performing department
            - Environment Satisfaction having a significant impact on performance across almost all departments.
            - Work Life Balance also having a noticeable impact on performane across almost all departments.
                    """)
        
    with st.expander("**Business Recommendations**"):

        st.markdown("""
            **Business Recommendations**

            - **Trial Flexible Working Arrangements** - Employee Work Life Balance was shown to have a sizeable impact on employee performance across almost all departments. It was also in the top ten most important features for predicting employee performance. It is recommended that INX trial flexible working arrangments such as flexi-hours or work-from-home where possible to help improve employee perceptions of work-life balance.
            - **Data Collection on "Low" rated Employees** - The dataset supplied for this analysis was notably absent of any data for employees on the lowest rank of performance ratings. This could be due to speedy retrenchment of said employees. However, this does create an analytical and modelling blindspot where the company is not aware of employees that could potentially have the biggest negative impact on the business through poor performance. It is recommended that in future data is collected on all employees regardless of performance rating level even if retrenchment is imminent.
            - **Review of Office Environment** - Employee Environment Satisfaction was the top-ranked factor influencing employee performance. It is recomemnded that a review be conducted into modernising or improving the office environment through measures such as updated decor, layout restructuring or potentially relocation. If this is coupled with the trialling or permanent implementation of flexible working arrangements, it could lead to a far smaller but higher quality office environment than is current in place.
            - **Internal Promotions** - Time worked at INX was in the top ten most important factors for predicting employee performance. It is recommended that preference be given to internal promotions over external hires for filling new vacancies where possible.
            - **Headcount Rebalancing** - The Finance department was the worst-performing of all the departments. Given how crucial proper financial management is to the survival of a business, it is recommended that a review be conducted into the possible automation with agentic AI agents or augmentation with AI tools such as large language models of certain roles in the three largest departments (Sales, Development and Research and Development) to potentially free up additional budget space for increased headcount in the Finance department. A concurrent review into the automation or AI augmentation of various finance functions should also be conducted. 
                    """)