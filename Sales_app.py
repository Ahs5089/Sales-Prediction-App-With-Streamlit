import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.image("IMG20230902154929.jpg") # TODO check this
# Load the model from a file
model_path = "model.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

scaler_path = "scalers.pkl"
with open(scaler_path, "rb") as model_file:
    model_scaler = pickle.load(model_file)
# Load the trained model and scaler files
# model = pickle.load(open('model.pkl', 'rb'))
# model_path = "D:/visual/New ads sales/model.pkl.pkl"
# with open(model_path, "rb") as model_file:
#     model = pickle.load(model_file)

#scaler = pickle.load(open("scaler.pkl", 'rb'))


# Create a Streamlit app and add input fields for TV, radio, and newspaper advertising
st.title('💰Sales Prediction')
st.write("*Developed for 🌎 with ❤️‍🔥 by Muhammad Ahsan👨🏻‍💻🇵🇰*")
st.write('Enter the advertising budget below to predict sales.')

st.sidebar.subheader("Contacts")
st.sidebar.write("""Shaikhahsan966@gmail.com,
+923103037663
""")

# Taking the input from user
company = st.text_input("Company Name") # TODO check this
new_tv = st.number_input('TV Advertising', min_value=0.0, max_value=1000.0, step=0.1)
new_radio = st.number_input('Radio Advertising', min_value=0.0, max_value=1000.0, step=0.1)
new_newspaper = st.number_input('Newspaper Advertising', min_value=0.0, max_value=1000.0, step=0.1)

# Button to trigger the prediction
if st.button('Predict'):
    new_value = pd.DataFrame([[new_tv, new_radio, new_newspaper]])
    new_value_scaled = model_scaler.transform(new_value)
    prediction = model.predict(new_value_scaled)
    st.markdown(f"Prediction result: **{prediction[0]}**")

