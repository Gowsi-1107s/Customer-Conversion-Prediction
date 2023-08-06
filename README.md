# CUSTOMER CONVERSION PREDICTION

# Problem-Statement
This abstract outlines a problem faced by a new-age insurance company that utilizes various outreach methods to sell term insurance to its customers. Despite employing multiple strategies, telephonic marketing campaigns remain highly effective but costly. The objective is to optimize the marketing campaign by identifying potential customers who are most likely to convert and subscribe to the insurance plan. To achieve this, a machine learning model will be built using historical marketing data, which will predict whether a client is likely to subscribe to the insurance, enabling targeted and cost-efficient telephonic outreach.

# Aim
The aim of this project is to develop a machine learning model using the historical marketing data of the new-age insurance company to predict the likelihood of a client subscribing to the term insurance plan. By accurately identifying potential customers who are most likely to convert, the model will enable the company to strategically target them via telephonic marketing campaigns, reducing costs and increasing the effectiveness of outreach efforts. Ultimately, the goal is to optimize the marketing process and enhance the overall conversion rate for the insurance company's term insurance products.

# Input Features
age (numeric)

job : type of job

marital : marital status

educational_qual : education status

call_type : contact communication type

day: last contact day of the month (numeric)

mon: last contact month of year

dur: last contact duration, in seconds (numeric)

num_calls: number of contacts performed during this campaign and for this client

prev_outcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success") ** Output variable (desired target):**

y --> has the client subscribed to the insurance?

# Requirements
* Python
* Machine Learning
* Scikit-learn
* Numpy
* Pandas
* Matplotlib
* Seaborn
* Jupyter Notebook

# Methods
* Exploratory Data Analysis (EDA)
* Data Collection
* Data Cleaning
* Feature Engineering
* Modelling

# Machine Learning Algorithm
* Logistic Regression
* Random Forest
* Decision Tree
* KNN
* XG_BOOST

# Workflow
1. Data Collection: Gather the historical marketing data from the insurance company's database or any relevant sources. This data should include information about customers, their interactions, and whether they subscribed to the insurance or not.

2. Data Exploration: Perform initial data exploration to understand the structure of the data, check for missing values, examine the distribution of features, and identify potential data quality issues.

3. Data Preprocessing: Clean and preprocess the data to handle missing values, outliers, and inconsistencies. Convert categorical features into numerical representations using techniques like one-hot encoding or label encoding.

4. Feature Engineering: Create additional relevant features based on domain knowledge or extract meaningful information from existing ones to improve the model's predictive capabilities.

5. Train-Test Split: Divide the preprocessed data into training and testing sets. The training set will be used to train the ML model, and the testing set will be used to evaluate its performance on unseen data.

6. Feature Selection: Identify the most relevant features that have the most significant impact on the prediction task. Feature selection can improve model efficiency and reduce overfitting.

7. Model Selection: Choose an appropriate ML algorithm for binary classification, such as Logistic Regression, Random Forest, Gradient Boosting, or Support Vector Machines (SVM), based on the problem complexity and dataset size.

8. Model Training: Train the selected ML algorithm using the training data. The model will learn the underlying patterns and relationships between features and the target variable (subscription status).

9. Model Evaluation: Evaluate the model's performance using evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC on the testing set to assess its predictive capabilities.

10. Model Deployment: Once the ML model has been trained and evaluated successfully, deploy it in a production environment. 
