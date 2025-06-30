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
# Diccionario estados conocidos y sus poblaciones
# -------------------------------------------------

states_dict = {
    'South Carolina': ('SC', 5190700),
    'New York': ('NY', 20201249),
    'Florida': ('FL', 21538187),
    'Michigan': ('MI', 10077331),
    'California': ('CA', 39538223),
    'Pennsylvania': ('PA', 13002700),
    'Texas': ('TX', 29145505),
    'Kentucky': ('KY', 4505836),
    'Wyoming': ('WY', 576851),
    'Alabama': ('AL', 5024279),
    'Louisiana': ('LA', 4657757),
    'Georgia': ('GA', 10711908),
    'Colorado': ('CO', 5773714),
    'Ohio': ('OH', 11799448),
    'Wisconsin': ('WI', 5893718),
    'Arkansas': ('AR', 3011524),
    'New Jersey': ('NJ', 9288994),
    'Iowa': ('IA', 3190369),
    'Maryland': ('MD', 6177224),
    'Mississippi': ('MS', 2961279),
    'Kansas': ('KS', 2937880),
    'Illinois': ('IL', 12812508),
    'Missouri': ('MO', 6154913),
    'Maine': ('ME', 1362359),
    'Tennessee': ('TN', 6910840),
    'Minnesota': ('MN', 5706494),
    'Oklahoma': ('OK', 3959353),
    'Washington': ('WA', 7693612),
    'West Virginia': ('WV', 1793716),
    'New Mexico': ('NM', 2117522),
    'Nebraska': ('NE', 1961504),
    'Virginia': ('VA', 8631393),
    'Oregon': ('OR', 4237256),
    'Indiana': ('IN', 6785528),
    'North Carolina': ('NC', 10439388),
    'North Dakota': ('ND', 779094)
}

# -------------------------------------------------
# Diccionario de TODAS las poblaciones USA (Censo 2020)
# -------------------------------------------------

all_states_pop = {
    'Alabama': 5024279,
    'Alaska': 733391,
    'Arizona': 7151502,
    'Arkansas': 3011524,
    'California': 39538223,
    'Colorado': 5773714,
    'Connecticut': 3605944,
    'Delaware': 989948,
    'Florida': 21538187,
    'Georgia': 10711908,
    'Hawaii': 1455271,
    'Idaho': 1839106,
    'Illinois': 12812508,
    'Indiana': 6785528,
    'Iowa': 3190369,
    'Kansas': 2937880,
    'Kentucky': 4505836,
    'Louisiana': 4657757,
    'Maine': 1362359,
    'Maryland': 6177224,
    'Massachusetts': 7029917,
    'Michigan': 10077331,
    'Minnesota': 5706494,
    'Mississippi': 2961279,
    'Missouri': 6154913,
    'Montana': 1084225,
    'Nebraska': 1961504,
    'Nevada': 3104614,
    'New Hampshire': 1377529,
    'New Jersey': 9288994,
    'New Mexico': 2117522,
    'New York': 20201249,
    'North Carolina': 10439388,
    'North Dakota': 779094,
    'Ohio': 11799448,
    'Oklahoma': 3959353,
    'Oregon': 4237256,
    'Pennsylvania': 13002700,
    'Rhode Island': 1097379,
    'South Carolina': 5190700,
    'South Dakota': 886667,
    'Tennessee': 6910840,
    'Texas': 29145505,
    'Utah': 3271616,
    'Vermont': 643077,
    'Virginia': 8631393,
    'Washington': 7693612,
    'West Virginia': 1793716,
    'Wisconsin': 5893718,
    'Wyoming': 576851
}

# Diccionario: lo que se muestra al usuario → lo que espera el modelo
category_display_dict = {
    "Otros - Internet": "misc_net",
    "Supermercado - Pago en tienda": "grocery_pos",
    "Entretenimiento": "entertainment",
    "Gasolina y Transporte": "gas_transport",
    "Otros - Pago en tienda": "misc_pos",
    "Supermercado - Online": "grocery_net",
    "Compras Online": "shopping_net",
    "Compras en Tienda": "shopping_pos",
    "Restaurantes y Comidas": "food_dining",
    "Cuidado Personal": "personal_care",
    "Salud y Fitness": "health_fitness",
    "Viajes": "travel",
    "Niños y Mascotas": "kids_pets",
    "Hogar": "home"
}

# -------------------------------------------------
# Entradas amigables
# -------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input(
        "💰 Transaction Amount (USD)", 
        min_value=0.0, max_value=1_000_000.0, value=1.0, step=5.0
    )
    gender_str = st.selectbox("👤 Gender", options=["Hombre", "Mujer"])
    age = st.number_input("🎂 Age", min_value=0, max_value=120, value=30, step=1)

with col2:
    state_str = st.selectbox("🌍 State of Transaction", options=sorted(list(all_states_pop.keys()) + ["Others"]))
    category_display = st.selectbox(
        "🗂️ Transaction Category",
        options=list(category_display_dict.keys())
    )
    category = category_display_dict[category_display]

with col3:
    weekday_name = st.selectbox(
        "📅 Weekday",
        options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    hour = st.slider("⏰ Hour of Transaction (0-23)", min_value=0, max_value=23, value=12, step=1)
    sector = st.selectbox(
        "🏢 Sector",
        options=[
            "Finanzas y Contabilidad",
            "Tecnología y Datos",
            "Ingeniería y Construcción",
            "Salud y Medicina",
            "Educación e Investigación",
            "Medios y Comunicación",
            "Consultoría y Asesoramiento",
            "Diseño y Arte",
            "Servicios Sociales y Comunitarios",
            "Administración y Gestión",
            "Ciencias Ambientales y Naturales",
            "Legal y Regulador",
            "Comercio y Ventas",
            "Hostelería y Turismo",
            "Transporte",
            "Otros"
        ]
    )

st.divider()

# -------------------------------------------------
# Transformaciones para predicción
# -------------------------------------------------

# Género: Hombre=1, Mujer=0
gender = 1 if gender_str == "Hombre" else 0

# Abreviatura y población
if state_str == "Others":
    state_abbr = "others"
    city_pop = 0
elif state_str in states_dict:
    state_abbr, city_pop = states_dict[state_str]
else:
    state_abbr = "others"
    city_pop = all_states_pop.get(state_str, 0)

# Hora a seno y coseno
hour_rad = 2 * math.pi * hour / 24
hour_sin = math.sin(hour_rad)
hour_cos = math.cos(hour_rad)

# -------------------------------------------------
# Botón para predecir
# -------------------------------------------------

if st.button("🔍 Detect Fraud"):

    # Creamos DataFrame con los nombres correctos
    input_df = pd.DataFrame([{
        "category": category,
        "amt": amount,
        "gender": gender,
        "state": state_abbr,
        "city_pop": city_pop,
        "weekday_name": weekday_name,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "age": age,
        "sector": sector
    }])

    # Cargamos el modelo
    with open("../models/finish_model.pkl", "rb") as archivo_salida:
        mi_modelo = pickle.load(archivo_salida)

    # Hacemos la predicción
    prediction = mi_modelo.predict(input_df)
    probability = mi_modelo.predict_proba(input_df)

    st.divider()

    if prediction[0] == 1:
        st.markdown(
            f"""
            <div style="background-color: #f8d7da; padding: 1rem; border-radius: 0.5rem; color: #721c24; font-weight: bold;">
                🚫 FRAUD DETECTED!<br>Estimated probability: {probability[0][1]*100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; color: #155724; font-weight: bold;">
                ✅ Transaction SAFE!<br>Estimated fraud probability: {probability[0][1]*100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

