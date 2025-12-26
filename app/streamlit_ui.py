import streamlit as st
import requests
import pandas as pd

st.markdown("<h1 style='text-align: center;'>🚇 Классификация текстов о метро</h1>", unsafe_allow_html=True)

# Выбор способа ввода
option = st.radio("Выберите способ:", ["Ввести тексты", "Загрузить CSV"])

if option == "Ввести тексты":
    texts = st.text_area("Тексты:", height=150)
    text_list = [t.strip() for t in texts.split("\n") if t.strip()] if texts else []
else:
    file = st.file_uploader("Загрузите CSV:", type="csv")
    text_list = pd.read_csv(file)['text'].tolist() if file else []

if text_list and st.button("Классифицировать"):
    # Отправка на API
    df = pd.DataFrame({"text": text_list})
    files = {"file": ("data.csv", df.to_csv(index=False), "text/csv")}
    
    response = requests.post("http://localhost:8000/predict", files=files)
    
    if response.status_code == 200:
        result = response.json()
        st.dataframe(pd.DataFrame(result['results']))
    else:
        st.error("Ошибка API")