#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay, accuracy_score, balanced_accuracy_score


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px


# ## 1.EDA
# 

# In[12]:


df_churn=pd.read_csv(r"C:\Users\krish\OneDrive\Desktop\churn.csv")
df_churn.head(5)


# In[13]:


df_churn.info()


# ### Tranform some of the binary features from numerical (0/1) to categorical for better interpretability and visualization.

# In[14]:


df_churn.rename(columns={'HasCrCard': 'Credit Card Status', 'IsActiveMember': 'Activity', 'Exited': 'Status'}, inplace=True)
df_churn.Status = df_churn.Status.map({0: 'Retained', 1: 'Exited'})
df_churn['Credit Card Status'] = df_churn['Credit Card Status'].map({0: 'No Credit Card', 1: 'Has Credit Card'})
df_churn.Activity = df_churn.Activity.map({0: 'Inactive', 1: 'Active'})
df_churn.head()


# In[15]:


# Check missing values
msno.matrix(df_churn)


# In[16]:


##Matrix :
##Using this matrix you can very quickly find the pattern of missingness in the dataset.
##No missing value.


# ### Let's look at the division of male and female customers.

# In[20]:



px.pie(df_churn, names='Gender', hole=0.5)


# In[21]:


px.pie(df_churn, names='Geography', hole=0.5)


# In[22]:


px.histogram(df_churn, x='Age', color='Status')


# In[23]:


px.pie(df_churn,names='Status',title='Percentage Churn', hole=0.5)


# In[24]:


px.histogram(df_churn, x='CreditScore', color='Status')


# In[25]:


px.box(df_churn, x='Tenure', color='Status')


# In[26]:


px.histogram(df_churn, x='Balance', color='Status')


# In[27]:


px.histogram(df_churn, x='EstimatedSalary', color='Status')


# In[28]:


fig = px.sunburst(df_churn, path=['Credit Card Status', 'Status'])
fig.update_traces(textinfo='label + percent parent')


# In[29]:


fig = px.sunburst(df_churn, path=['Activity', 'Status'])
fig.update_traces(textinfo='label + percent parent')


# In[30]:


fig = px.sunburst(df_churn, path=['NumOfProducts', 'Status'])
fig.update_traces(textinfo='label + percent parent')


# ## 2 Preprocessing and Feature engineering

# In[31]:


#Drop unnecessary columns
df_churn.columns


# In[32]:


df_churn.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
df_churn.head()


# In[33]:


# Need to handle the categorical features. For this, we create a binary encoding for each class in each categorical feature.
pd.get_dummies(df_churn)


# In[34]:


X = df_churn.iloc[:, :-1]
X = pd.get_dummies(X)
X.head()


# In[35]:


y = df_churn.Status


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## 3 Logistic Regression Model

# In[37]:


#Training
lr_model = LogisticRegression(penalty='l2', tol=0.0001, C=1.0, fit_intercept=True, class_weight='balanced',
                              random_state=42, solver='lbfgs', max_iter=100, verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
# Fit to train data
lr_model.fit(X_train, y_train)


# In[38]:


y_pred = lr_model.predict(X_test)


# In[44]:


# Calculate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(confusion, display_labels=lr_model.classes_).plot(cmap='Blues', ax=ax, values_format=".2f")
plt.show()



# In[ ]:





# In[ ]:





# In[47]:




# Create a figure and axis for the ROC curve plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the ROC curve for the logistic regression model
plot_roc_curve(lr_model, X_test, y_test, ax=ax)

# Add a diagonal line for reference
ax.plot([0, 1], [0, 1], 'k--')

# Set the plot title
fig.suptitle('ROC Curve for LR Model')

# Show the ROC curve plot
plt.show()


# In[48]:


y_pred_prob = lr_model.predict_proba(X_test) 
roc_auc_score(y_test, y_pred_prob[:, 1])


# ### While the current classifier exhibits some level of accuracy, it may not be a strong performer. We have the option to investigate more complex and advanced models that could potentially yield improved classification results.

# ## 4 Decision Tree Model

# In[49]:


# Initialize model with parameters
dt_model = DecisionTreeClassifier(max_depth=30, min_samples_split=50, min_samples_leaf=25, 
                                  max_leaf_nodes=100, class_weight='balanced', ccp_alpha=0.0001)
# Fit model to training data
dt_model.fit(X_train, y_train)


# In[50]:


# Plot tree
plt.figure(figsize=(15, 10))
tree.plot_tree(dt_model, filled=True)
plt.show()


# In[51]:


# Use trained model to make predictions on test data
y_pred = dt_model.predict(X_test)


# In[54]:


#Calculate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Create a confusion matrix display
disp = ConfusionMatrixDisplay(confusion, display_labels=lr_model.classes_)

# Plot the confusion matrix with decimal precision and normalize it
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap='Blues', ax=ax, values_format=".2f", include_values=True, xticks_rotation='horizontal')

# Show the plot
plt.show()


# In[56]:


# Create a figure and axis for the ROC curve plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the ROC curve for the decision tree model
plot_roc_curve(dt_model, X_test, y_test, pos_label='Exited', ax=ax)

# Add a diagonal line for reference
ax.plot([0, 1], [0, 1], 'k--')

# Set the plot title
fig.suptitle('ROC Curve for DT Model')

# Show the ROC curve plot
plt.show()


# In[57]:


# Calculate area under roc curve
y_pred_prob = dt_model.predict_proba(X_test) 
roc_auc_score(y_test, y_pred_prob[:, 1])


# In[59]:


feature_importances = dt_model.feature_importances_
feature_names = X.columns  # Replace with your actual feature names

# Create a DataFrame with feature names and importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Create a bar chart using Plotly Express
fig = px.bar(importance_df, x='Feature', y='Importance')

# Customize the appearance of the chart (e.g., axis labels, title)
fig.update_layout(title='Feature Importances for Decision Tree Model', xaxis_title='Feature', yaxis_title='Importance')

# Show the plot
fig.show()


# ## The most important feature being age, followed by number of products subscribed and balance.

# In[ ]:




