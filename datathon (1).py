#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importing necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical computations
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.ensemble import RandomForestClassifier  # For building a random forest classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # For model evaluation


# In[5]:


df=pd.read_csv("Training.csv")
df


# In[6]:


df.dtypes


# In[7]:


df.isnull().sum()


# In[8]:


categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols)
print(df_encoded.head())


# In[9]:


from sklearn.preprocessing import StandardScaler

numerical_features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 
                      'no_of_week_nights', 'required_car_parking_space', 'lead_time',
                      'arrival_year', 'arrival_month', 'arrival_date',
                      'repeated_guest', 'no_of_previous_cancellations',
                      'no_of_previous_bookings_not_canceled', 'avg_price_per_room',
                      'no_of_special_requests']

# Apply Standardization to numerical features
scaler = StandardScaler() 
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Assuming 'df' is the original DataFrame containing the data before encoding
# Extracting the target variable 'booking_status' from the original DataFrame
y = df['booking_status']

# Assuming 'X' is the DataFrame containing the features after one-hot encoding and z-score normalization
X = df_encoded.drop(columns=['Booking_ID'])  # Drop non-numeric and non-predictive columns

# Splitting the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Printing the shapes of training and testing setsy0
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# Assuming X_train and y_train are the training features and target variable
# Based on the dataset characteristics and the need for high accuracy, we'll choose Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Evaluating model performance using cross-validation
# We'll use cross-validation to get a more reliable estimate of the model's performance
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)  # 5-fold cross-validation

# Calculating the mean cross-validation score
mean_cv_score = cv_scores.mean()

# Print the mean cross-validation score
print("Mean Cross-Validation Score for Random Forest Classifier:", mean_cv_score)


# In[11]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Assuming 'X_train' and 'y_train' are the training features and target variable
# Assuming 'X_test' and 'y_test' are the testing features and target variable

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train (fit) the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Now, the classifier is fitted and ready to make predictions
# Predict the labels for the testing data
y_pred = rf_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", conf_matrix)


# In[12]:


pip install gradio


# In[26]:


import pickle

# Train your Random Forest Classifier and obtain the model 'rf_classifier'

# Save the model to a file using pickle
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)


# In[6]:


import gradio as gr
import joblib

# Load the trained model
rf_classifier = joblib.load('random_forest_model.pkl')
def predict(no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
            type_of_meal_plan, required_car_parking_space, room_type_reserved, lead_time,
            arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest,
            no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
            avg_price_per_room, no_of_special_requests):

    # Convert categorical features to one-hot encoding
    type_of_meal_plan_encoded = 1 if type_of_meal_plan == 'yes' else 0
    room_type_reserved_encoded = 1 if room_type_reserved == 'yes' else 0
    market_segment_type_encoded = 1 if market_segment_type == 'yes' else 0

    # Predict with the loaded model
    features = [no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
                required_car_parking_space, lead_time, arrival_year, arrival_month,
                arrival_date, repeated_guest, no_of_previous_cancellations,
                no_of_previous_bookings_not_canceled, avg_price_per_room,
                no_of_special_requests, type_of_meal_plan_encoded,
                room_type_reserved_encoded, market_segment_type_encoded]
    prediction = rf_classifier.predict([features])[0]
    return prediction
inputs = [
    gr.inputs.Number(label="Number of Adults"),
    gr.inputs.Number(label="Number of Children"),
    gr.inputs.Number(label="Number of Weekend Nights"),
    gr.inputs.Number(label="Number of Week Nights"),
    gr.inputs.Dropdown(["yes", "no"], label="Type of Meal Plan"),
    gr.inputs.Number(label="Required Car Parking Space"),
    gr.inputs.Dropdown(["yes", "no"], label="Room Type Reserved"),
    gr.inputs.Number(label="Lead Time"),
    gr.inputs.Number(label="Arrival Year"),
    gr.inputs.Number(label="Arrival Month"),
    gr.inputs.Number(label="Arrival Date"),
    gr.inputs.Dropdown(["yes", "no"], label="Market Segment Type"),
    gr.inputs.Number(label="Repeated Guest"),
    gr.inputs.Number(label="Number of Previous Cancellations"),
    gr.inputs.Number(label="Number of Previous Bookings Not Canceled"),
    gr.inputs.Number(label="Average Price Per Room"),
    gr.inputs.Number(label="Number of Special Requests")
]
output = gr.outputs.Label(label="Booking Status Prediction")
gr.Interface(predict, inputs, output, title="Booking Status Predictor").launch()


# In[ ]:





# In[ ]:




