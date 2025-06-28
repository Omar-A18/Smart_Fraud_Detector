import streamlit as st
import numpy as np
import pandas as pd
import math
import pickle

# -------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -------------------------------------------------

st.set_page_config(
    page_title="Smart Fraud Detector",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------------------------
# TÍTULO Y DESCRIPCIÓN
# -------------------------------------------------

st.title("🛡️ Smart Fraud Detector")

st.markdown("""
Welcome to your Smart Fraud Detector in the USA.  
Enter your transaction to check for fraud.
""")

st.divider()

# -------------------------------------------------
# Diccionario estados
# -------------------------------------------------

states_dict = {
    'South Carolina': 'SC',
    'Others': 'others',
    'New York': 'NY',
    'Florida': 'FL',
    'Michigan': 'MI',
    'California': 'CA',
    'Pennsylvania': 'PA',
    'Texas': 'TX',
    'Kentucky': 'KY',
    'Wyoming': 'WY',
    'Alabama': 'AL',
    'Louisiana': 'LA',
    'Georgia': 'GA',
    'Colorado': 'CO',
    'Ohio': 'OH',
    'Wisconsin': 'WI',
    'Arkansas': 'AR',
    'New Jersey': 'NJ',
    'Iowa': 'IA',
    'Maryland': 'MD',
    'Mississippi': 'MS',
    'Kansas': 'KS',
    'Illinois': 'IL',
    'Missouri': 'MO',
    'Maine': 'ME',
    'Tennessee': 'TN',
    'Minnesota': 'MN',
    'Oklahoma': 'OK',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'New Mexico': 'NM',
    'Nebraska': 'NE',
    'Virginia': 'VA',
    'Oregon': 'OR',
    'Indiana': 'IN',
    'North Carolina': 'NC',
    'North Dakota': 'ND'
}

# -------------------------------------------------
# Entradas con labels amigables y transformaciones
# -------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input(
        "💰 Transaction Amount (USD)", 
        min_value=0.0, max_value=1_000_000.0, value=0.01, step=0.01
    )
    gender_str = st.selectbox("👤 Gender", options=["Hombre", "Mujer"])
    age = st.number_input("🎂 Age", min_value=0, max_value=120, value=30, step=1)

with col2:
    state_str = st.selectbox("🌍 State of Transaction", options=list(states_dict.keys()))
    city_pop = st.number_input(
        "🏙️ City Population", min_value=0, max_value=10_000_000, value=1000, step=1
    )
    category = st.selectbox(
        "🗂️ Transaction Category",
        options=[
            "misc_net", "grocery_pos", "entertainment", "gas_transport", "misc_pos", "food_dining"
        ]
    )

with col3:
    weekday_name = st.selectbox(
        "📅 Weekday",
        options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    hour = st.slider("⏰ Hour of Transaction (0-23)", min_value=0, max_value=23, value=12, step=1)
    sector = st.selectbox(
        "🏢 Sector",
        options=[
            "Servicios Sociales y Comunitarios",
            "Educación e Investigación",
            "Legal y Regulador",
            "other",
            "Ciencias Ambientales y Naturales",
            "Medios y Comunicación",
            "Ingeniería y Construcción"
        ]
    )

st.divider()

# -------------------------------------------------
# Transformaciones para predicción
# -------------------------------------------------

# Género: Hombre=1, Mujer=0
gender = 1 if gender_str == "Hombre" else 0

# Estado abreviado para modelo, others si no está
state = states_dict.get(state_str, "others")

# Codificación weekday_name (ejemplo simple)
weekday_map = {
    "Monday": 0, "Tuesday":1, "Wednesday":2, "Thursday":3,
    "Friday":4, "Saturday":5, "Sunday":6
}
weekday = weekday_map.get(weekday_name, 0)

# Convertir hora a seno y coseno para variable circular
hour_rad = 2 * math.pi * hour / 24
hour_sin = math.sin(hour_rad)
hour_cos = math.cos(hour_rad)


# Aquí irían otras codificaciones o transformaciones según modelo

# -------------------------------------------------
# Botón para predecir
# -------------------------------------------------

if st.button("🔍 Detect Fraud"):

    # Preparar datos en orden correcto para tu modelo
    # Ejemplo (ajusta según las variables que use tu modelo):
    input_data = np.array([[
        amount,
        gender,
        state,
        city_pop,
        category,
        weekday,
        hour_sin,
        hour_cos,
        age,
        sector,
        device
    ]], dtype=object)  # Usa object si hay mezcla de tipos

    # Aquí tendrías que transformar categóricas a números si tu modelo no las acepta strings
    # Por ejemplo, si usas OneHotEncoding o LabelEncoding, hazlo aquí

    # Cargar modelo
    with open("finish_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Predecir
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.divider()

    if prediction[0] == 1:
        st.error(f"🚫 **FRAUD DETECTED!**\n\nProbabilidad estimada: {probability[0][1]*100:.2f}%")
    else:
        st.success(f"✅ **Transaction SAFE!**\n\nProbabilidad estimada de fraude: {probability[0][1]*100:.2f}%")

    st.divider()

    # Mostrar resumen datos usados
    summary = pd.DataFrame({
        "Variable": [
            "Amount", "Gender", "State", "City Population", "Category", "Weekday",
            "Hour Sin", "Hour Cos", "Age", "Sector", "Device"
        ],
        "Value": [
            amount, gender_str, state, city_pop, category, weekday_name,
            f"{hour_sin:.3f}", f"{hour_cos:.3f}", age, sector, device_str
        ]
    })

    st.markdown("#### Transaction Details")
    st.table(summary)
