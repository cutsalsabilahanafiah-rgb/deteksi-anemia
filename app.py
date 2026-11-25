import streamlit as st
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Skrining Anemia Konjungtiva", layout="centered")
st.title("Skrining Dini Anemia via Konjungtiva")
st.write("Unggah foto kelopak mata bagian bawah (konjungtiva) dengan pencahayaan terang.")

model = joblib.load("model_anemia.pkl")

file_foto = st.file_uploader("Pilih atau ambil foto konjungtiva", type=["jpg", "jpeg", "png"])

if file_foto is not None:
    gambar = Image.open(file_foto).convert("RGB")
    st.image(gambar, caption="Foto yang diunggah", use_container_width=True)
    
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
            st.error(f"**Hasil Skrining: TERINDIKASI ANEMIA (Yes)** — Tingkat Keyakinan: {peluang:.1f}%")
        else:
            st.success(f"**Hasil Skrining: NORMAL / TIDAK ANEMIA (No)** — Tingkat Keyakinan: {peluang:.1f}%")
            
        st.caption(f"Komposisi Warna -> Red: {red_pct:.1f}%, Green: {green_pct:.1f}%, Blue: {blue_pct:.1f}%")
