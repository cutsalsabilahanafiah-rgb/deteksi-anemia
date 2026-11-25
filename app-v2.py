import streamlit as st
import numpy as np
import joblib
from PIL import Image

st.set_page_config(
    page_title="Skrining Dini Anemia",
    page_icon="🩸",
    layout="centered"
)

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
        font-size: 26px;
    }
    .header-box p {
        color: #FFEBEE !important;
        margin: 0;
        font-size: 15px;
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
        <h2>🩸 Skrining Dini Anemia via Konjungtiva</h2>
        <p>Analisis komposisi warna kelopak mata bagian bawah secara cepat berbasis <i>Machine Learning</i></p>
    </div>
""", unsafe_allow_html=True)

with st.expander("📸 Panduan Mengambil Foto Konjungtiva agar Hasil Akurat"):
    st.markdown("""
    * **Pencahayaan:** Gunakan cahaya terang (cahaya matahari atau lampu putih ruangan).
    * **Posisi:** Tarik perlahan kelopak mata bagian bawah hingga area dalam (konjungtiva) terlihat jelas.
    * **Pemotongan (Crop):** Potong foto agar **hanya fokus pada bagian dalam kelopak mata yang kemerahan** (hindari ikut terpotongnya kulit wajah, bulu mata, atau jari).
    """)

model = joblib.load("model_anemia.pkl")

file_foto = st.file_uploader("📤 Unggah Foto Konjungtiva (Format: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if file_foto is not None:
    gambar = Image.open(file_foto).convert("RGB")
    st.divider()
    
    kolom_kiri, kolom_kanan = st.columns([1, 1.3])
    
    with kolom_kiri:
        st.image(gambar, caption="Foto Konjungtiva", use_container_width=True)
        
    with kolom_kanan:
        st.subheader("📋 Hasil Analisis")
        img_array = np.array(gambar)
        r_mean, g_mean, b_mean = np.mean(img_array, axis=(0, 1))
        total = r_mean + g_mean + b_mean
        
        if total > 0:
            red_pct = (r_mean / total) * 100
            green_pct = (g_mean / total) * 100
            blue_pct = (b_mean / total) * 100
            
            fitur = np.array([[red_pct, green_pct, blue_pct]])
            prediksi = model.predict(fitur)
            peluang = np.max(model.predict_proba(fitur)) * 100
            
            if prediksi == "Yes":
                st.error("🚨 **TERINDIKASI ANEMIA**")
                st.write("Komposisi warna konjungtiva cenderung pucat. Sebaiknya segera lakukan pemeriksaan kadar Hemoglobin (Hb) lebih lanjut.")
            else:
                st.success("✅ **NORMAL (TIDAK TERINDIKASI ANEMIA)**")
                st.write("Komposisi warna konjungtiva berada dalam kategori normal. Tetap jaga pola makan bergizi dan asupan zat besi.")
                
            st.metric(label="Tingkat Keyakinan Model", value=f"{peluang:.1f}%")
            
            with st.expander("Lihat Detail Persentase Warna Pixel"):
                st.write(f"🔴 **Red Pixel:** {red_pct:.2f}%")
                st.write(f"🟢 **Green Pixel:** {green_pct:.2f}%")
                st.write(f"🔵 **Blue Pixel:** {blue_pct:.2f}%")

st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Pengingat Penting (Disclaimer Medis):</strong><br>
        Layanan ini merupakan alat <strong>skrining awal (deteksi dini)</strong> berbasis kecerdasan buatan dan 
        <strong>bukan pengganti pemeriksaan dokter, tenaga kesehatan, maupun diagnosis laboratorium</strong>. 
        Hasil analisis dapat dipengaruhi oleh kualitas kamera dan pencahayaan ruangan. Untuk kepastian diagnosis 
        dan penanganan medis yang tepat, silakan berkonsultasi langsung dengan dokter atau fasilitas pelayanan kesehatan terdekat.
    </div>
""", unsafe_allow_html=True)
