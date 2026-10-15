
import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

st.set_page_config(
    page_title="Regresi Kota Makassar",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Regresi Kota Makassar")
st.caption("Analisis indikator fiskal dan pembangunan tahun 2014–2024")

# Membaca dataset dari GitHub
@st.cache_data
def load_data():
    return pd.read_csv("dataset_fiskal.csv")

try:
    data = load_data()
except Exception as e:
    st.error("Dataset tidak ditemukan. Pastikan dataset_fiskal.csv ada di repository GitHub.")
    st.stop()

# Filter Kota Makassar
df = data[data["Kota"].str.strip().str.lower() == "kota makassar"].copy()
df = df.sort_values("Tahun")

if df.empty:
    st.error("Data Kota Makassar tidak ditemukan.")
    st.stop()

# Sidebar
st.sidebar.title("Pengaturan Analisis")
tahun = st.sidebar.slider(
    "Rentang Tahun",
    int(df["Tahun"].min()),
    int(df["Tahun"].max()),
    (int(df["Tahun"].min()), int(df["Tahun"].max()))
)

df = df[df["Tahun"].between(tahun[0], tahun[1])].copy()

st.subheader("Ringkasan Data")

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Observasi", len(df))
c2.metric("Rata-rata IPM", f"{df['IPM'].mean():.2f}")
c3.metric("Rata-rata Pertumbuhan Ekonomi", f"{df['PE'].mean():.2f}")

st.divider()

# Grafik tren
kiri, kanan = st.columns(2)

with kiri:
    st.subheader("Tren IPM")
    fig = px.line(
        df, x="Tahun", y="IPM",
        markers=True,
        title="Perubahan IPM Kota Makassar"
    )
    st.plotly_chart(fig, use_container_width=True)

with kanan:
    st.subheader("Tren Pendapatan Asli Daerah")
    fig = px.line(
        df, x="Tahun", y="PAD",
        markers=True,
        title="Perubahan PAD Kota Makassar"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Dataset Kota Makassar")
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Regresi Linear Berganda")

# Pilih variabel
angka = [
    c for c in df.select_dtypes(include="number").columns
    if c != "Tahun"
]

variabel_y = st.selectbox(
    "Variabel dependen (Y)",
    angka,
    index=angka.index("IPM") if "IPM" in angka else 0
)

pilihan_x = [c for c in angka if c != variabel_y]
variabel_x = st.multiselect(
    "Variabel independen (X)",
    pilihan_x,
    default=[
        c for c in ["PAD", "ICT"]
        if c in pilihan_x
    ]
)

if st.button("Jalankan Regresi", type="primary"):
    if len(variabel_x) == 0:
        st.warning("Pilih minimal satu variabel X.")
    elif len(df) <= len(variabel_x) + 1:
        st.warning("Observasi terlalu sedikit untuk jumlah variabel X.")
    else:
        model_data = df[[variabel_y] + variabel_x].dropna()

        if len(model_data) <= len(variabel_x) + 1:
            st.warning("Data valid tidak cukup untuk menjalankan model.")
        else:
            X = sm.add_constant(model_data[variabel_x])
            y = model_data[variabel_y]

            model = sm.OLS(y, X).fit()

            a, b, c = st.columns(3)
            a.metric("R-squared", f"{model.rsquared:.4f}")
            b.metric("Adjusted R-squared", f"{model.rsquared_adj:.4f}")
            c.metric("Prob (F-statistic)", f"{model.f_pvalue:.4f}")

            st.subheader("Koefisien Regresi")
            hasil = pd.DataFrame({
                "Koefisien": model.params,
                "P-value": model.pvalues
            })
            st.dataframe(hasil, use_container_width=True)

            st.subheader("Ringkasan Model")
            st.code(model.summary().as_text())

            st.subheader("Nilai Aktual dan Prediksi")
            hasil_prediksi = model_data.copy()
            hasil_prediksi["Prediksi"] = model.predict(X)
            st.dataframe(hasil_prediksi, use_container_width=True)

            st.caption(
                "Interpretasi perlu mempertimbangkan jumlah observasi, "
                "asumsi model, dan keterbatasan data. "
                "Korelasi atau regresi tidak membuktikan sebab-akibat."
            )
