
import streamlit as st

st.set_page_config(page_title="Aplikasi Saya")

st.title("Aplikasi Streamlit Saya")
st.write("Halo! Aplikasi saya berhasil dijalankan.")
st.success("Streamlit berhasil terhubung dengan GitHub!")

import streamlit as st
import pandas as pd
import statsmodels.api as sm

st.title("Analisis Regresi Linear Berganda")

data = {
    "PMakan": [50, 40, 60, 55, 45, 65, 70, 75, 80, 90],
    "PTranspor": [20, 25, 30, 35, 40, 50, 55, 60, 65, 70],
    "BB": [60, 55, 65, 70, 62, 75, 80, 85, 90, 95]
}

df = pd.DataFrame(data)

st.subheader("Dataset")
st.dataframe(df)

X = df[["PMakan", "PTranspor"]]
y = df["BB"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

st.subheader("Hasil Regresi")
st.text(model.summary().as_text())
