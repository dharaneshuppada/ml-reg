import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Pre-trained model (you can directly use the trained LinearRegression model here)
lr_regressor = LinearRegression()

# Label encoders dictionary for encoding categorical inputs
def load_label_encoders():
    encoders = {}
    # Add label encoding for the columns used during training
    encoders['Gender'] = LabelEncoder().fit(['Male', 'Female'])
    encoders['Location'] = LabelEncoder().fit(['Pakistan', 'Mexico', 'United States', 'Barzil'])
    encoders['Platform'] = LabelEncoder().fit(['Instagram', 'Facebook', 'YouTube', 'TikTok'])
    encoders['Demographics'] = LabelEncoder().fit(['Urban', 'Rural'])
    encoders['Profession'] = LabelEncoder().fit(['Engineer', 'Artist', 'Manager', 'Waiting staff'])
    encoders['DeviceType'] = LabelEncoder().fit(['Smartphone', 'Tablet', 'Computer'])
    encoders['OS'] = LabelEncoder().fit(['Android', 'iOS'])
    encoders['Watch Reason'] = LabelEncoder().fit(['Procrastination', 'Habit', 'Entertainment', 'Boredom'])
    encoders['CurrentActivity'] = LabelEncoder().fit(['Commuting', 'At school', 'At home'])
    encoders['ConnectionType'] = LabelEncoder().fit(['Mobile Data', 'Wi-Fi'])
    return encoders

label_encoders = load_label_encoders()

# Function to get user input
def user_input_features():
    age = st.slider('Age', 18, 100, 25)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    location = st.selectbox('Location', ['Pakistan', 'Mexico', 'United States', 'Barzil'])
    income = st.number_input('Income', min_value=0, max_value=100000, value=30000)
    debt = st.checkbox('Debt')
    owns_property = st.checkbox('Owns Property')
    profession = st.selectbox('Profession', ['Engineer', 'Artist', 'Manager', 'Waiting staff'])
    platform = st.selectbox('Social Media Platform', ['Instagram', 'Facebook', 'YouTube', 'TikTok'])
    total_time_spent = st.slider('Total Time Spent (minutes)', 0, 1000, 100)
    number_of_sessions = st.slider('Number of Sessions', 1, 100, 10)
    engagement = st.slider('Engagement Score', 1, 10, 5)
    productivity_loss = st.slider('Productivity Loss', 0, 10, 5)
    satisfaction = st.slider('Satisfaction Score', 1, 10, 5)
    watch_reason = st.selectbox('Reason for Watching', ['Procrastination', 'Habit', 'Entertainment', 'Boredom'])
    device_type = st.selectbox('Device Type', ['Smartphone', 'Tablet', 'Computer'])
    os = st.selectbox('Operating System', ['Android', 'iOS'])
    current_activity = st.selectbox('Current Activity', ['Commuting', 'At school', 'At home'])
    connection_type = st.selectbox('Connection Type', ['Mobile Data', 'Wi-Fi'])

    # Store inputs into a dictionary
    user_data = {
        'Age': age,
        'Gender': label_encoders['Gender'].transform([gender])[0],
        'Location': label_encoders['Location'].transform([location])[0],
        'Income': income,
        'Debt': int(debt),
        'Owns Property': int(owns_property),
        'Profession': label_encoders['Profession'].transform([profession])[0],
        'Platform': label_encoders['Platform'].transform([platform])[0],
        'Total Time Spent': total_time_spent,
        'Number of Sessions': number_of_sessions,
        'Engagement': engagement,
        'ProductivityLoss': productivity_loss,
        'Satisfaction': satisfaction,
        'Watch Reason': label_encoders['Watch Reason'].transform([watch_reason])[0],
        'DeviceType': label_encoders['DeviceType'].transform([device_type])[0],
        'OS': label_encoders['OS'].transform([os])[0],
        'CurrentActivity': label_encoders['CurrentActivity'].transform([current_activity])[0],
        'ConnectionType': label_encoders['ConnectionType'].transform([connection_type])[0],
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

# Streamlit app
st.title('Simple Social Media Addiction Predictor')

# Get user input
input_df = user_input_features()

# Show the input data
st.write("Your Input Data:")
st.dataframe(input_df)

# Model prediction
if st.button('Predict Addiction Level'):
    prediction = lr_regressor.predict(input_df)
    st.write(f"Predicted Addiction Level: {prediction[0]:.2f}")
