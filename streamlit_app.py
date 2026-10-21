
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from pathlib import Path

# =========================================================
# 1. KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Makassar Insight",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. TEMA DAN CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #F3F8FF 0%, #FFFFFF 48%);
    color: #172B4D;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1F3A 0%, #123E70 100%);
}

[data-testid="stSidebar"] * {
    color: #F3F7FF !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #172B4D !important;
}

[data-testid="stSidebar"] input {
    color: #172B4D !important;
}

.hero {
    background: linear-gradient(120deg, #0B1F3A 0%, #125AA3 65%, #2589D8 100%);
    padding: 30px 34px;
    border-radius: 22px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 12px 30px rgba(18, 73, 132, 0.15);
}

.hero h1 {
    color: white;
    font-size: 35px;
    font-weight: 800;
    margin: 0 0 8px 0;
}

.hero p {
    color: #DCEBFF;
    font-size: 15px;
    margin: 0;
}

.section-title {
    color: #123E70;
    font-size: 22px;
    font-weight: 750;
    margin: 18px 0 12px 0;
}

.metric-card {
    background: white;
    padding: 21px;
    border: 1px solid #DFEAF7;
    border-radius: 17px;
    box-shadow: 0 5px 18px rgba(22, 70, 120, 0.06);
    min-height: 120px;
}

.metric-label {
    color: #647A96;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 9px;
}

.metric-value {
    color: #123E70;
    font-size: 27px;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.metric-note {
    color: #7185A0;
    font-size: 12px;
    margin-top: 7px;
}

.info-card {
    background: #EDF5FF;
    border-left: 4px solid #2589D8;
    padding: 15px 18px;
    border-radius: 10px;
    color: #23466D;
    margin: 10px 0 18px 0;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #DFEAF7;
    padding: 17px;
    border-radius: 15px;
    box-shadow: 0 4px 14px rgba(22, 70, 120, 0.04);
}

.stButton > button, .stDownloadButton > button {
    background: #1769AA;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    background: #0B1F3A;
    color: white;
    border: none;
}

h1, h2, h3 {
    color: #123E70;
}

hr {
    border-color: #DCE8F5;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. MEMBACA DATASET
# =========================================================
NAMA_FILE = "dataset_fiskal_97_hipoFull (1).csv"

@st.cache_data
def load_data():
    # Mencari file pada folder aplikasi
    path = Path(NAMA_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"File {NAMA_FILE} tidak ditemukan. "
            "Pastikan file CSV berada satu folder dengan streamlit_app.py."
        )

    df = pd.read_csv(path)

    # Merapikan nama kolom
    df.columns = df.columns.astype(str).str.strip()

    # Merapikan data nama kota
    if "Kota" in df.columns:
        df["Kota"] = df["Kota"].astype("string").str.strip()

    # Mengubah kolom angka yang terbaca sebagai teks
    for col in df.columns:
        if col not in ["Kota"]:
            if df[col].dtype == "object":
                cleaned = (
                    df[col].astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.strip()
                )
                converted = pd.to_numeric(cleaned, errors="coerce")

                if converted.notna().sum() > 0:
                    df[col] = converted

    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Data gagal dibaca: {e}")
    st.info(
        "Periksa nama file CSV, lokasi file, dan pastikan "
        "file sudah diunggah ke repository GitHub."
    )
    st.stop()

if df.empty:
    st.error("Dataset tidak memiliki baris data.")
    st.stop()

if "Kota" not in df.columns:
    st.error("Kolom 'Kota' tidak ditemukan dalam dataset.")
    st.write("Kolom yang tersedia:", df.columns.tolist())
    st.stop()

# Memastikan kolom angka dikenali
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.error("Tidak ditemukan kolom numerik untuk dianalisis.")
    st.stop()

kota_list = sorted(
    df["Kota"].dropna().astype(str).unique().tolist()
)

makassar_list = [
    kota for kota in kota_list
    if "makassar" in kota.lower()
]

if not makassar_list:
    st.error("Kota Makassar tidak ditemukan dalam dataset.")
    st.write("Nama kota yang tersedia:", kota_list)
    st.stop()

KOTA_UTAMA = makassar_list[0]

# =========================================================
# 4. FUNGSI BANTUAN
# =========================================================
BLUE = "#1769AA"
NAVY = "#123E70"
LIGHT_BLUE = "#62A9E8"
PALETTE = ["#1769AA", "#62A9E8", "#8CC8F5",
           "#315A85", "#A6D7FA", "#4B83B5"]

def format_angka(nilai):
    if pd.isna(nilai):
        return "N/A"
    return f"{nilai:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")

def buat_grafik(fig, tinggi=400):
    fig.update_layout(
        height=tinggi,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", color="#34506F"),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(
        gridcolor="#E8EFF7",
        zerolinecolor="#D8E5F3"
    )
    return fig

def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def filter_tahun(data):
    if "Tahun" not in data.columns:
        return data.copy()

    tahun_valid = pd.to_numeric(data["Tahun"], errors="coerce").dropna()

    if tahun_valid.empty:
        return data.copy()

    tahun_list = sorted(tahun_valid.unique().tolist())

    if len(tahun_list) == 1:
        st.sidebar.caption(f"Tahun data: {tahun_list[0]:.0f}")
        return data[data["Tahun"] == tahun_list[0]].copy()

    tahun_awal, tahun_akhir = st.sidebar.select_slider(
        "Rentang Tahun",
        options=tahun_list,
        value=(tahun_list[0], tahun_list[-1]),
        format_func=lambda x: str(int(x))
    )

    tahun_numeric = pd.to_numeric(data["Tahun"], errors="coerce")

    return data[
        tahun_numeric.between(tahun_awal, tahun_akhir)
    ].copy()

# =========================================================
# 5. SIDEBAR DAN FILTER
# =========================================================
with st.sidebar:
    st.markdown("## 🌊 MAKASSAR")
    st.caption("INSIGHT • DATA • ANALYTICS")
    st.divider()

    halaman = st.radio(
        "NAVIGASI DASHBOARD",
        [
            "Beranda",
            "Perbandingan Kota",
            "Eksplorasi Data",
            "Analisis Regresi",
            "Data & Unduhan"
        ]
    )

    st.divider()
    st.markdown("### ⚙️ Pengaturan Data")

    data_tahun = filter_tahun(df)

    st.caption(f"Total kota dalam dataset: {len(kota_list)}")
    st.caption(f"Jumlah variabel numerik: {len(numeric_cols)}")

# Data Makassar sesuai filter tahun
data_makassar = data_tahun[
    data_tahun["Kota"].astype(str).str.lower() == KOTA_UTAMA.lower()
].copy()

if data_makassar.empty:
    data_makassar = df[
        df["Kota"].astype(str).str.lower() == KOTA_UTAMA.lower()
    ].copy()

# =========================================================
# 6. HEADER
# =========================================================
st.markdown(f"""
<div class="hero">
    <h1>Makassar Insight</h1>
    <p>Dashboard analisis data, indikator pembangunan, dan
    perbandingan antar kota berbasis data.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 7. HALAMAN BERANDA
# =========================================================
if halaman == "Beranda":

    st.markdown('<div class="section-title">Ringkasan Dashboard</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Total Observasi",
            format_angka(len(data_tahun)),
            "Baris data sesuai filter"
        )

    with col2:
        metric_card(
            "Jumlah Kota",
            str(data_tahun["Kota"].nunique()),
            "Kota yang tercatat"
        )

    with col3:
        metric_card(
            "Periode Data",
            (
                f"{int(data_tahun['Tahun'].min())}–"
                f"{int(data_tahun['Tahun'].max())}"
                if "Tahun" in data_tahun.columns
                and data_tahun["Tahun"].notna().any()
                else "Tidak tersedia"
            ),
            "Rentang tahun terpilih"
        )

    with col4:
        metric_card(
            "Data Makassar",
            str(len(data_makassar)),
            "Observasi Kota Makassar"
        )

    st.markdown('<div class="section-title">Tren Indikator Makassar</div>',
                unsafe_allow_html=True)

    pilihan_indikator = st.selectbox(
        "Pilih indikator yang ingin ditampilkan",
        numeric_cols,
        index=numeric_cols.index("IPM") if "IPM" in numeric_cols else 0
    )

    if "Tahun" in data_makassar.columns and not data_makassar.empty:
        tren = data_makassar.dropna(
            subset=["Tahun", pilihan_indikator]
        ).sort_values("Tahun")

        if not tren.empty:
            fig = px.line(
                tren,
                x="Tahun",
                y=pilihan_indikator,
                markers=True,
                title=f"Perkembangan {pilihan_indikator} di {KOTA_UTAMA}",
                color_discrete_sequence=[BLUE]
            )
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=8)
            )
            st.plotly_chart(buat_grafik(fig, 430), use_container_width=True)
        else:
            st.info("Belum ada data lengkap untuk membuat grafik tren.")
    else:
        st.info("Grafik tren membutuhkan kolom Tahun.")

    col_a, col_b = st.columns([1.15, 1])

    with col_a:
        st.markdown("#### Data Terbaru Makassar")

        terbaru = data_makassar.copy()

        if "Tahun" in terbaru.columns:
            terbaru = terbaru.sort_values("Tahun", ascending=False)

        st.dataframe(
            terbaru.head(8),
            use_container_width=True,
            hide_index=True
        )

    with col_b:
        st.markdown("#### Statistik Deskriptif")

        statistik = data_makassar[numeric_cols].describe().T

        statistik = statistik.rename(columns={
            "count": "Jumlah",
            "mean": "Rata-rata",
            "std": "Std. Deviasi",
            "min": "Minimum",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Maksimum"
        })

        st.dataframe(
            statistik.round(2),
            use_container_width=True
        )

# =========================================================
# 8. HALAMAN PERBANDINGAN KOTA
# =========================================================
elif halaman == "Perbandingan Kota":

    st.markdown('<div class="section-title">Perbandingan Antar Kota</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    Bandingkan posisi Kota Makassar dengan kota lain berdasarkan
    indikator dan rentang tahun yang dipilih.
    </div>
    """, unsafe_allow_html=True)

    kota_pembanding = st.multiselect(
        "Pilih kota untuk dibandingkan",
        options=kota_list,
        default=[KOTA_UTAMA],
        max_selections=12
    )

    indikator = st.selectbox(
        "Indikator perbandingan",
        numeric_cols,
        index=numeric_cols.index("IPM") if "IPM" in numeric_cols else 0
    )

    if kota_pembanding:
        perbandingan = data_tahun[
            data_tahun["Kota"].astype(str).isin(kota_pembanding)
        ].copy()

        if perbandingan.empty:
            st.warning("Tidak ada data untuk pilihan tersebut.")
        else:
            ringkasan = (
                perbandingan.groupby("Kota", as_index=False)[indikator]
                .mean()
                .dropna()
                .sort_values(indikator, ascending=False)
            )

            col1, col2 = st.columns([1.4, 1])

            with col1:
                fig = px.bar(
                    ringkasan,
                    x="Kota",
                    y=indikator,
                    color="Kota",
                    title=f"Rata-rata {indikator} per Kota",
                    color_discrete_sequence=PALETTE,
                    text_auto=".2f"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(
                    buat_grafik(fig, 430),
                    use_container_width=True
                )

            with col2:
                st.markdown("#### Peringkat Berdasarkan Rata-rata")
                ranking = ringkasan.reset_index(drop=True)
                ranking.index = ranking.index + 1
                ranking.index.name = "Peringkat"
                st.dataframe(
                    ranking,
                    use_container_width=True
                )

            if "Tahun" in perbandingan.columns:
                tren_banding = (
                    perbandingan.groupby(
                        ["Tahun", "Kota"], as_index=False
                    )[indikator].mean()
                )

                fig_tren = px.line(
                    tren_banding,
                    x="Tahun",
                    y=indikator,
                    color="Kota",
                    markers=True,
                    title=f"Tren {indikator} Antar Kota",
                    color_discrete_sequence=PALETTE
                )
                st.plotly_chart(
                    buat_grafik(fig_tren, 420),
                    use_container_width=True
                )

            st.markdown("#### Tabel Perbandingan")
            st.dataframe(
                ringkasan,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.info("Pilih setidaknya satu kota untuk memulai perbandingan.")

# =========================================================
# 9. HALAMAN EKSPLORASI DATA
# =========================================================
elif halaman == "Eksplorasi Data":

    st.markdown('<div class="section-title">Eksplorasi Data</div>',
                unsafe_allow_html=True)

    sumber = st.radio(
        "Pilih cakupan data",
        ["Kota Makassar", "Semua Kota"],
        horizontal=True
    )

    data_eksplorasi = (
        data_makassar.copy()
        if sumber == "Kota Makassar"
        else data_tahun.copy()
    )

    col1, col2 = st.columns(2)

    with col1:
        kolom_analisis = st.selectbox(
            "Pilih variabel untuk analisis",
            numeric_cols,
            index=numeric_cols.index("IPM") if "IPM" in numeric_cols else 0
        )

    with col2:
        tipe_grafik = st.selectbox(
            "Jenis visualisasi",
            ["Histogram", "Box Plot", "Scatter Plot"]
        )

    if tipe_grafik == "Histogram":
        fig = px.histogram(
            data_eksplorasi,
            x=kolom_analisis,
            nbins=25,
            title=f"Distribusi {kolom_analisis}",
            color_discrete_sequence=[BLUE]
        )

    elif tipe_grafik == "Box Plot":
        fig = px.box(
            data_eksplorasi,
            y=kolom_analisis,
            title=f"Sebaran dan Pencilan {kolom_analisis}",
            color_discrete_sequence=[LIGHT_BLUE]
        )

    else:
        kandidat_x = [c for c in numeric_cols if c != kolom_analisis]

        if kandidat_x:
            variabel_x = st.selectbox(
                "Variabel sumbu X",
                kandidat_x
            )

            fig = px.scatter(
                data_eksplorasi,
                x=variabel_x,
                y=kolom_analisis,
                color="Kota" if sumber == "Semua Kota" else None,
                hover_data=["Tahun"] if "Tahun" in data_eksplorasi.columns else None,
                title=f"Hubungan {variabel_x} dan {kolom_analisis}",
                color_discrete_sequence=PALETTE,
                trendline="ols" if len(data_eksplorasi) > 2 else None
            )
        else:
            fig = None
            st.info("Dibutuhkan minimal dua variabel numerik.")

    if fig is not None:
        st.plotly_chart(
            buat_grafik(fig, 430),
            use_container_width=True
        )

    st.markdown("#### Matriks Korelasi")

    korelasi = data_eksplorasi[numeric_cols].corr()

    fig_corr = px.imshow(
        korelasi,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[
            "#F0F7FF", "#8CC8F5", "#1769AA", "#0B1F3A"
        ],
        title="Korelasi Antarvariabel"
    )

    st.plotly_chart(
        buat_grafik(fig_corr, 500),
        use_container_width=True
    )

    st.caption(
        "Korelasi menunjukkan arah dan kekuatan hubungan linear, "
        "bukan bukti sebab-akibat."
    )

    st.markdown("#### Pemeriksaan Kualitas Data")

    kualitas = pd.DataFrame({
        "Variabel": data_eksplorasi.columns,
        "Tipe Data": [
            str(data_eksplorasi[c].dtype)
            for c in data_eksplorasi.columns
        ],
        "Data Kosong": [
            int(data_eksplorasi[c].isna().sum())
            for c in data_eksplorasi.columns
        ],
        "Persentase Kosong (%)": [
            round(data_eksplorasi[c].isna().mean() * 100, 2)
            for c in data_eksplorasi.columns
        ]
    })

    st.dataframe(
        kualitas,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# 10. HALAMAN ANALISIS REGRESI
# =========================================================
elif halaman == "Analisis Regresi":

    st.markdown('<div class="section-title">Analisis Regresi Linear</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    Gunakan regresi untuk mempelajari hubungan antara satu variabel
    target dan satu atau lebih variabel penjelas. Hasilnya bergantung
    pada kualitas data dan tidak otomatis membuktikan hubungan sebab-akibat.
    </div>
    """, unsafe_allow_html=True)

    data_regresi = data_makassar.copy()

    target_default = "IPM" if "IPM" in numeric_cols else numeric_cols[0]

    target = st.selectbox(
        "Variabel target (Y)",
        numeric_cols,
        index=numeric_cols.index(target_default)
    )

    kandidat_x = [c for c in numeric_cols if c != target]

    default_x = [
        c for c in ["DAK_Nonfisik", "DBH_SDA"]
        if c in kandidat_x
    ]

    if not default_x and kandidat_x:
        default_x = kandidat_x[:min(2, len(kandidat_x))]

    prediktor = st.multiselect(
        "Variabel penjelas (X)",
        kandidat_x,
        default=default_x
    )

    if not prediktor:
        st.warning("Pilih setidaknya satu variabel penjelas.")
    else:
        kolom_model = [target] + prediktor

        model_data = data_regresi[kolom_model].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        st.write(f"Jumlah observasi yang digunakan: **{len(model_data)}**")

        if len(model_data) <= len(prediktor) + 1:
            st.warning(
                "Observasi tidak cukup untuk membentuk model yang stabil. "
                "Tambahkan data atau kurangi jumlah variabel X."
            )
        else:
            X = sm.add_constant(model_data[prediktor], has_constant="add")
            y = model_data[target]

            try:
                model = sm.OLS(y, X).fit()

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "R-squared",
                        f"{model.rsquared:.4f}"
                    )

                with c2:
                    st.metric(
                        "Adjusted R-squared",
                        f"{model.rsquared_adj:.4f}"
                    )

                with c3:
                    st.metric(
                        "Observasi",
                        f"{int(model.nobs)}"
                    )

                st.markdown("#### Ringkasan Hasil Model")
                st.text(model.summary().as_text())

                koef = pd.DataFrame({
                    "Variabel": model.params.index,
                    "Koefisien": model.params.values,
                    "P-value": model.pvalues.values
                })

                st.markdown("#### Tabel Koefisien")
                st.dataframe(
                    koef.round(5),
                    use_container_width=True,
                    hide_index=True
                )

                hasil_prediksi = pd.DataFrame({
                    "Aktual": y,
                    "Prediksi": model.predict(X)
                })

                fig_reg = px.scatter(
                    hasil_prediksi,
                    x="Aktual",
                    y="Prediksi",
                    title="Nilai Aktual vs Prediksi",
                    color_discrete_sequence=[BLUE]
                )

                nilai_min = min(
                    hasil_prediksi["Aktual"].min(),
                    hasil_prediksi["Prediksi"].min()
                )
                nilai_max = max(
                    hasil_prediksi["Aktual"].max(),
                    hasil_prediksi["Prediksi"].max()
                )

                fig_reg.add_trace(go.Scatter(
                    x=[nilai_min, nilai_max],
                    y=[nilai_min, nilai_max],
                    mode="lines",
                    name="Garis ideal",
                    line=dict(color="#8CC8F5", dash="dash")
                ))

                st.plotly_chart(
                    buat_grafik(fig_reg, 430),
                    use_container_width=True
                )

                st.caption(
                    "R-squared mengukur proporsi variasi target yang "
                    "dijelaskan model pada data yang dianalisis. "
                    "Periksa juga p-value, ukuran sampel, dan asumsi model."
                )

            except Exception as e:
                st.error(f"Model regresi gagal dibuat: {e}")

# =========================================================
# 11. HALAMAN DATA DAN UNDUHAN
# =========================================================
elif halaman == "Data & Unduhan":

    st.markdown('<div class="section-title">Data & Unduhan</div>',
                unsafe_allow_html=True)

    cakupan = st.radio(
        "Data yang ditampilkan",
        ["Makassar", "Semua Kota"],
        horizontal=True
    )

    data_tampil = (
        data_makassar.copy()
        if cakupan == "Makassar"
        else data_tahun.copy()
    )

    pencarian = st.text_input(
        "Cari isi data",
        placeholder="Contoh: Makassar atau nama indikator"
    )

    if pencarian:
        mask = data_tampil.astype(str).apply(
            lambda kolom: kolom.str.contains(
                pencarian, case=False, na=False
            )
        ).any(axis=1)

        data_tampil = data_tampil[mask]

    st.write(f"Menampilkan **{len(data_tampil)} baris data**")

    st.dataframe(
        data_tampil,
        use_container_width=True,
        hide_index=True
    )

    csv = data_tampil.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Unduh Data CSV",
        data=csv,
        file_name="data_makassar_insight.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("#### Informasi Dataset")

    c1, c2, c3 = st.columns(3)

    c1.metric("Jumlah Baris", len(df))
    c2.metric("Jumlah Kolom", len(df.columns))
    c3.metric("Data Kosong", int(df.isna().sum().sum()))

# =========================================================
# 12. FOOTER
# =========================================================
st.divider()

st.markdown("""
<div style="text-align:center; padding:10px 0 20px 0;">
    <span style="color:#1769AA; font-weight:800;">
        MAKASSAR INSIGHT
    </span>
    <br>
    <span style="color:#7185A0; font-size:12px;">
        Dashboard eksplorasi dan analisis data berbasis Python
    </span>
</div>
""", unsafe_allow_html=True)
