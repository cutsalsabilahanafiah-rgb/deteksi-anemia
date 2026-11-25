import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Skrining Dini Obesitas & Pola Hidup",
    page_icon="⚖️",
    layout="centered"
)

# Desain Merah-Putih yang Selaras dengan Google Sites
st.markdown("""
    <style>
    .header-box {
        background: linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%);
        padding: 24px;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(211, 47, 47, 0.18);
    }
    .header-box h2 {
        color: white !important;
        margin: 0 0 8px 0;
        font-size: 25px;
    }
    .header-box p {
        color: #FFEBEE !important;
        margin: 0;
        font-size: 14px;
    }
    .disclaimer-box {
        background-color: #FFF5F5;
        border-left: 5px solid #D32F2F;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 25px;
        color: #333333;
        font-size: 14px;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-box">
        <h2>⚖️ Self-Assessment Risiko Obesitas & Pola Hidup</h2>
        <p>Evaluasi status berat badan dan kebiasaan harian berbasis <i>Machine Learning</i> & Standar Kemenkes RI</p>
    </div>
""", unsafe_allow_html=True)

# Muat Model
model = joblib.load("model_obesitas.pkl")

st.subheader("1️⃣ Data Fisik & Riwayat")
col1, col2 = st.columns(2)
with col1:
    gender_teks = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    usia = st.number_input("Usia (Tahun)", min_value=14, max_value=80, value=20)
    riwayat_teks = st.selectbox("Riwayat Keluarga Kelebihan Berat Badan?", ["Tidak", "Ya"])

with col2:
    tinggi_cm = st.number_input("Tinggi Badan (cm)", min_value=120.0, max_value=220.0, value=160.0, step=0.5)
    berat_kg = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=55.0, step=0.5)

st.divider()
st.subheader("2️⃣ Kebiasaan Makan & Aktivitas Harian")
col3, col4 = st.columns(2)
with col3:
    favc_teks = st.selectbox("Sering makan makanan tinggi kalori/cepat saji?", ["Tidak", "Ya"])
    fcvc_teks = st.selectbox("Frekuensi konsumsi sayur dalam makanan", ["Jarang (1x)", "Kadang-kadang (2x)", "Selalu / Setiap Makan (3x)"])
    ch2o_teks = st.selectbox("Konsumsi air putih harian", ["Kurang dari 1 Liter", "1 – 2 Liter", "Lebih dari 2 Liter"])

with col4:
    faf_teks = st.selectbox("Frekuensi olahraga / aktivitas fisik per minggu", ["Tidak pernah", "1 – 2 hari/minggu", "3 – 4 hari/minggu", "5 hari atau lebih/minggu"])
    tue_teks = st.selectbox("Durasi menatap layar (HP/Laptop/TV) per hari", ["0 – 2 jam/hari", "3 – 5 jam/hari", "Lebih dari 5 jam/hari"])

st.write("")
tombol_cek = st.button("🔍 Analisis Risiko Sekarang", type="primary", use_container_width=True)

if tombol_cek:
    # Konversi input pengguna ke angka sesuai format dataset
    gender = 1 if gender_teks == "Laki-laki" else 0
    riwayat = 1 if riwayat_teks == "Ya" else 0
    favc = 1 if favc_teks == "Ya" else 0
    
    map_fcvc = {"Jarang (1x)": 1.0, "Kadang-kadang (2x)": 2.0, "Selalu / Setiap Makan (3x)": 3.0}
    map_ch2o = {"Kurang dari 1 Liter": 1.0, "1 – 2 Liter": 2.0, "Lebih dari 2 Liter": 3.0}
    map_faf = {"Tidak pernah": 0.0, "1 – 2 hari/minggu": 1.0, "3 – 4 hari/minggu": 2.0, "5 hari atau lebih/minggu": 3.0}
    map_tue = {"0 – 2 jam/hari": 0.0, "3 – 5 jam/hari": 1.0, "Lebih dari 5 jam/hari": 2.0}
    
    tinggi_m = tinggi_cm / 100.0
    fitur_input = np.array([[
        gender, usia, tinggi_m, berat_kg, riwayat, favc,
        map_fcvc[fcvc_teks], map_ch2o[ch2o_teks], map_faf[faf_teks], map_tue[tue_teks]
    ]])
    
    # Prediksi Machine Learning (ambil nilai teks tunggalnya)
    hasil_ml = str(model.predict(fitur_input).item())
    peluang = float(np.max(model.predict_proba(fitur_input))) * 100
    
    terjemahan_label = {
        "Insufficient_Weight": ("Berat Badan Kurang (Underweight)", "info"),
        "Normal_Weight": ("Berat Badan Normal", "success"),
        "Overweight_Level_I": ("Kelebihan Berat Badan Tingkat I (Overweight I)", "warning"),
        "Overweight_Level_II": ("Kelebihan Berat Badan Tingkat II (Overweight II)", "warning"),
        "Obesity_Type_I": ("Obesitas Tipe I", "error"),
        "Obesity_Type_II": ("Obesitas Tipe II", "error"),
        "Obesity_Type_III": ("Obesitas Tipe III (Berat)", "error")
    }
    label_indo, warna_kotak = terjemahan_label.get(hasil_ml, (hasil_ml, "info"))
    
    # Hitung IMT Standar Kemenkes RI (Asia-Pasifik)
    imt = berat_kg / (tinggi_m ** 2)
    if imt < 18.5:
        kat_kemenkes = "Kurus (Kekurangan Berat Badan)"
    elif imt <= 22.9:
        kat_kemenkes = "Normal"
    elif imt <= 24.9:
        kat_kemenkes = "Overweight (Berisiko Obesitas)"
    else:
        kat_kemenkes = "Obesitas"
        
    st.divider()
    st.subheader("📋 Hasil Self-Assessment")
    
    if warna_kotak == "success":
        st.success(f"**Prediksi Pola Hidup & Fisik (ML):** {label_indo} (Keyakinan Model: {peluang:.1f}%)")
    elif warna_kotak == "warning":
        st.warning(f"**Prediksi Pola Hidup & Fisik (ML):** {label_indo} (Keyakinan Model: {peluang:.1f}%)")
    elif warna_kotak == "error":
        st.error(f"**Prediksi Pola Hidup & Fisik (ML):** {label_indo} (Keyakinan Model: {peluang:.1f}%)")
    else:
        st.info(f"**Prediksi Pola Hidup & Fisik (ML):** {label_indo} (Keyakinan Model: {peluang:.1f}%)")
        
    c1, c2 = st.columns(2)
    c1.metric("Nilai IMT Kamu", f"{imt:.1f} kg/m²")
    c2.metric("Kategori Standar Kemenkes RI", kat_kemenkes)

st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Pengingat Penting (Disclaimer Medis):</strong><br>
        Layanan ini merupakan alat <strong>skrining awal (self-assessment)</strong> berbasis kecerdasan buatan yang 
        memadukan pola gaya hidup dengan standar ambang batas IMT Kemenkes RI, dan <strong>bukan pengganti pemeriksaan dokter, 
        ahli gizi, maupun tenaga kesehatan</strong>. Untuk perencanaan diet dan penanganan medis yang tepat, 
        silakan berkonsultasi langsung dengan tenaga kesehatan di fasilitas pelayanan kesehatan terdekat.
    </div>
""", unsafe_allow_html=True)
