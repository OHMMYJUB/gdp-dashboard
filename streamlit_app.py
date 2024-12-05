import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = load_model('diabetes_model.h5')  # หรือระบุเส้นทางของไฟล์โมเดลของคุณ

# สร้างฟังก์ชั่นสำหรับการทำนาย
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    # สร้าง DataFrame จากข้อมูลที่ป้อนมา
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # สเกลข้อมูลที่กรอก (ใช้ StandardScaler หรือ MinMaxScaler)
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # ทำนายผลจากโมเดล
    prediction = model.predict(input_data_scaled)
    return prediction[0][0]  # คืนค่าผลลัพธ์ทำนาย (0 หรือ 1)

# UI สำหรับการกรอกข้อมูล
st.title("Diabetes Prediction Web App")
st.write("กรุณากรอกข้อมูลเพื่อทำนายโรคเบาหวาน")

# รับข้อมูลจากผู้ใช้
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20)
glucose = st.number_input('Glucose Level', min_value=0, max_value=300)
blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200)
skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100)
insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0)
age = st.number_input('Age', min_value=1, max_value=120)

# คำนวณผลเมื่อกดปุ่ม
if st.button('Predict'):
    result = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    if result == 1:
        st.write('ผลการทำนาย: คุณอาจจะเป็นโรคเบาหวาน')
    else:
        st.write('ผลการทำนาย: คุณไม่เป็นโรคเบาหวาน')
