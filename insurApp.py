import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import io

# Load the dataset (assuming the dataset has been previously trained and saved)
data = pd.read_csv('personalized_insurance_dataset.csv')

# Initialize label encoders and encode categorical variables
label_encoders = {}
categorical_cols = ['occupation', 'marital_status', 'customer_preferences', 'policy_recommendation']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save label encoders for later use

# Define features and target variable
X = data[['age', 'income', 'occupation', 'marital_status', 'children', 'risk_score', 'customer_preferences']]
y = data['policy_recommendation']

# Train the best model
best_model = RandomForestClassifier()
best_model.fit(X, y)

# Function to make predictions and provide explanations
def make_predictions(input_data):
    predictions = best_model.predict(input_data)
    # Inverse transform the encoded labels for policy recommendations
    policy_le = label_encoders['policy_recommendation']
    predicted_labels = policy_le.inverse_transform(predictions)
    
    explanations = []
    for index, row in input_data.iterrows():
        risk = row['risk_score']
        advice = ""
        if risk > 0.8:
            advice = "High risk score; consider comprehensive coverage."
        elif risk > 0.5:
            advice = "Moderate risk score; balance cost with coverage."
        else:
            advice = "Low risk score; budget-friendly options may suffice."
        explanations.append(advice)
    return predicted_labels, explanations


# Main application
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Home", "Policy Recommender"])

    # Home page
    if page == "Home":
        st.title("Welcome to InsurApp")
        st.write(
            "Welcome to **InsurApp**, a predictive analytics solution for cost savings and increased returns on investment."
        )
    
    # Policy Recommender page
    elif page == "Policy Recommender":
        st.title("Insurance Policy Recommendation")

        # User input method
        input_option = st.radio("Choose input method:", ('Fill Form', 'Upload CSV/Excel'))

        if input_option == 'Fill Form':
            # Display form to get input from user
            st.header("Enter Customer Details")
            
            with st.form(key='insurance_form'):
                age = st.number_input('Age', min_value=18, max_value=100, value=30)
                income = st.number_input('Annual Income', min_value=20000, max_value=200000, value=50000)
                
                occupation_options = label_encoders['occupation'].classes_
                occupation = st.selectbox('Occupation', occupation_options)
                
                marital_status_options = label_encoders['marital_status'].classes_
                marital_status = st.selectbox('Marital Status', marital_status_options)
                
                children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
                
                risk_score = st.slider('Risk Score (0 - 1)', min_value=0.0, max_value=1.0, value=0.5)
                
                customer_preferences_options = label_encoders['customer_preferences'].classes_
                customer_preferences = st.selectbox('Customer Preferences', customer_preferences_options)
                
                # Predict and Cancel buttons
                predict_button = st.form_submit_button(label='Predict')
                #cancel_button = st.form_submit_button(label='Cancel')

            #if cancel_button:
                # Reset to initial state by setting query parameters
                #st.experimental_set_query_params()

            if predict_button:
                # Prepare input data
                input_data = pd.DataFrame({
                    'age': [age],
                    'income': [income],
                    'occupation': [occupation],
                    'marital_status': [marital_status],
                    'children': [children],
                    'risk_score': [risk_score],
                    'customer_preferences': [customer_preferences]
                })
                
                # Encode the input data
                for col in ['occupation', 'marital_status', 'customer_preferences']:
                    le = label_encoders[col]
                    input_data[col] = le.transform(input_data[col])
                
                # Make prediction
                prediction, advice = make_predictions(input_data)
                
                st.subheader("Prediction Result")
                st.write(f"**Recommended Policy**: {prediction[0]}")
                st.write(f"**Advice**: {advice[0]}")

        elif input_option == 'Upload CSV/Excel':
            st.header("Upload Customer Data File")
            
            # Provide template download
            st.markdown("Download a template file to populate customer data:")
            template_df = pd.DataFrame({
                'age': [],
                'income': [],
                'occupation': [],  # Placeholder for users to input
                'marital_status': [],  # Placeholder for users to input
                'children': [],
                'risk_score': [],
                'customer_preferences': []  # Placeholder for users to input
            })
            
            # Create CSV template
            csv_template = template_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Template",
                data=csv_template,
                file_name='insurance_template.csv',
                mime='text/csv'
            )
            
            # Create Excel template
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False)
            st.download_button(
                label="Download Excel Template",
                data=excel_buffer.getvalue(),
                file_name='insurance_template.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            
            # File uploader
            uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    input_df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    input_df = pd.read_excel(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    st.stop()
                
                # Check if required columns are present
                required_columns = ['age', 'income', 'occupation', 'marital_status', 'children', 'risk_score', 'customer_preferences']
                if not all(column in input_df.columns for column in required_columns):
                    st.error(f"The uploaded file must contain the following columns: {', '.join(required_columns)}")
                    st.stop()
                
                # Encode the input data
                for col in ['occupation', 'marital_status', 'customer_preferences']:
                    le = label_encoders[col]
                    input_df[col] = le.transform(input_df[col])
                
                # Predict and Cancel buttons
                predict_button = st.button('Predict')
                cancel_button = st.button('Cancel')
                
                if cancel_button:
                    st.experimental_set_query_params()

                if predict_button:
                    # Make predictions
                    predictions, advices = make_predictions(input_df)
                    input_df['policy_recommendation'] = predictions
                    input_df['advice'] = advices

                    st.subheader("Prediction Results")
                    st.write(input_df.head())  # Show first 5 records

                    # Option to download the results
                    result_csv = input_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=result_csv,
                        file_name='predictions.csv',
                        mime='text/csv'
                    )
                    
                    result_excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(result_excel_buffer, engine='openpyxl') as writer:
                        input_df.to_excel(writer, index=False)
                    st.download_button(
                        label="Download Predictions as Excel",
                        data=result_excel_buffer.getvalue(),
                        file_name='predictions.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

# Run the app
if __name__ == "__main__":
    main()
