
import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

# =====================================
# 1. PENGATURAN HALAMAN
# =====================================
st.set_page_config(
    page_title="Regresi Kota Makassar",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CityInsight: Analisis Kota Makassar")
st.write(
    "Dashboard eksplorasi data fiskal, ekonomi, "
    "dan pembangunan Kota Makassar."
)

# =====================================
# 2. MEMBACA DATASET
# =====================================
@st.cache_data
def load_data():
    return pd.read_csv("dataset_fiskal_97_hipoFull (1).csv")

try:
    data = load_data()
except Exception as e:
    st.error("File CSV gagal dibaca. Periksa nama file di GitHub.")
    st.exception(e)
    st.stop()

# =====================================
# 3. FILTER KOTA MAKASSAR
# =====================================
df = data[
    data["Kota"].astype(str).str.strip().str.lower()
    == "kota makassar"
].copy()

df = df.sort_values("Tahun")

if df.empty:
    st.error("Data Kota Makassar tidak ditemukan.")
    st.stop()

# =====================================
# 4. SIDEBAR DAN NAVIGASI
# =====================================
st.sidebar.title("🏙️ CityInsight")

halaman = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Beranda",
        "Eksplorasi Data",
        "Regresi Linear",
        "Tentang Data"
    ]
)

tahun_min = int(df["Tahun"].min())
tahun_max = int(df["Tahun"].max())

rentang = st.sidebar.slider(
    "Rentang Tahun",
    min_value=tahun_min,
    max_value=tahun_max,
    value=(tahun_min, tahun_max)
)

df = df[df["Tahun"].between(rentang[0], rentang[1])].copy()

# =====================================
# 5. HALAMAN BERANDA
# =====================================
if halaman == "Beranda":

    st.subheader("Ringkasan Data Kota Makassar")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Jumlah Data", len(df))
    col2.metric("Rata-rata IPM", f"{df['IPM'].mean():.2f}")
    col3.metric("Rata-rata PAD", f"{df['PAD'].mean():.2f}")
    col4.metric("Rata-rata Ekonomi", f"{df['PE'].mean():.2f}")

    st.divider()

    kiri, kanan = st.columns(2)

    with kiri:
        st.subheader("Tren IPM")

        fig = px.line(
            df,
            x="Tahun",
            y="IPM",
            markers=True,
            title="Perubahan IPM Kota Makassar"
        )

        st.plotly_chart(fig, use_container_width=True)

    with kanan:
        st.subheader("Tren Pendapatan Asli Daerah")

        fig = px.line(
            df,
            x="Tahun",
            y="PAD",
            markers=True,
            title="Perubahan PAD Kota Makassar"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Kota Makassar")
    st.dataframe(df, use_container_width=True, hide_index=True)

# =====================================
# 6. HALAMAN EKSPLORASI DATA
# =====================================
elif halaman == "Eksplorasi Data":

    st.subheader("Eksplorasi Variabel")

    kolom_numerik = [
        c for c in df.select_dtypes(include="number").columns
        if c != "Tahun"
    ]

    x_var = st.selectbox(
        "Pilih variabel X",
        kolom_numerik,
        index=kolom_numerik.index("PAD")
        if "PAD" in kolom_numerik else 0
    )

    y_var = st.selectbox(
        "Pilih variabel Y",
        kolom_numerik,
        index=kolom_numerik.index("IPM")
        if "IPM" in kolom_numerik else 0
    )

    fig = px.scatter(
        df,
        x=x_var,
        y=y_var,
        hover_data=["Tahun"],
        trendline="ols",
        title=f"Hubungan {x_var} dan {y_var}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(
        df[kolom_numerik].describe().T,
        use_container_width=True
    )

    st.download_button(
        "📥 Unduh Data Makassar",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="data_makassar.csv",
        mime="text/csv"
    )

# =====================================
# 7. HALAMAN REGRESI LINEAR
# =====================================
elif halaman == "Regresi Linear":

    st.subheader("Regresi Linear Berganda")

    st.write(
        "Pilih variabel yang ingin dianalisis. "
        "Contoh: IPM sebagai Y, PAD dan ICT sebagai X."
    )

    numerik = [
        c for c in df.select_dtypes(include="number").columns
        if c != "Tahun"
    ]

    y_var = st.selectbox(
        "Variabel dependen (Y)",
        numerik,
        index=numerik.index("IPM")
        if "IPM" in numerik else 0
    )

    pilihan_x = [c for c in numerik if c != y_var]

    x_vars = st.multiselect(
        "Variabel independen (X)",
        pilihan_x,
        default=[c for c in ["PAD", "ICT"] if c in pilihan_x]
    )

    if st.button("Jalankan Regresi", type="primary"):

        if len(x_vars) == 0:
            st.warning("Pilih minimal satu variabel X.")

        elif len(df) <= len(x_vars) + 1:
            st.warning(
                "Jumlah data terlalu sedikit untuk "
                "jumlah variabel yang dipilih."
            )

        else:
            analisis = df[[y_var] + x_vars].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()

            if len(analisis) <= len(x_vars) + 1:
                st.warning("Data valid tidak cukup untuk regresi.")

            else:
                X = sm.add_constant(analisis[x_vars])
                y = analisis[y_var]

                model = sm.OLS(y, X).fit()

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "R-squared",
                    f"{model.rsquared:.4f}"
                )

                c2.metric(
                    "Adjusted R-squared",
                    f"{model.rsquared_adj:.4f}"
                )

                c3.metric(
                    "P-value Uji F",
                    f"{model.f_pvalue:.4f}"
                )

                st.subheader("Koefisien Regresi")

                hasil = pd.DataFrame({
                    "Koefisien": model.params,
                    "P-value": model.pvalues
                })

                st.dataframe(hasil, use_container_width=True)

                st.subheader("Persamaan Regresi")

                persamaan = f"{y_var} = {model.params.iloc[0]:.4f}"

                for variabel in x_vars:
                    koef = model.params[variabel]
                    tanda = "+" if koef >= 0 else "-"
                    persamaan += f" {tanda} {abs(koef):.4f}({variabel})"

                st.code(persamaan)

                st.subheader("Ringkasan Hasil Statistik")
                st.text(model.summary().as_text())

                st.subheader("Aktual vs Prediksi")

                prediksi = analisis.copy()
                prediksi["Prediksi"] = model.predict(X)

                st.dataframe(
                    prediksi,
                    use_container_width=True
                )

                st.caption(
                    "Perhatikan jumlah observasi, asumsi regresi, "
                    "dan kemungkinan tren waktu. Hasil regresi "
                    "tidak otomatis membuktikan sebab-akibat."
                )

# =====================================
# 8. HALAMAN TENTANG DATA
# =====================================
elif halaman == "Tentang Data":

    st.subheader("Tentang Dataset")

    st.write(
        "Dataset memuat indikator fiskal, ekonomi, "
        "dan pembangunan untuk sejumlah kota dan tahun."
    )

    st.write("**Kota yang dianalisis:** Kota Makassar")
    st.write(f"**Tahun tersedia:** {tahun_min}–{tahun_max}")
    st.write(f"**Jumlah observasi pada filter:** {len(df)}")
    st.write(f"**Jumlah kolom dataset:** {len(data.columns)}")

    st.write("**Daftar variabel:**")
    st.write(", ".join(data.columns.tolist()))

    st.info(
        "Periksa dokumentasi sumber data untuk mengetahui "
        "definisi, satuan, dan metode pengukuran setiap variabel."
    )
