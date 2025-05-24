import streamlit as st
import os
from detect_custom import run_detection
import pandas as pd

st.set_page_config(page_title="Deteksi Hilal", layout="centered")

st.title("🌙 Aplikasi Deteksi Hilal Menggunakan YOLOv5")

uploaded_file = st.file_uploader("Unggah Gambar atau Video", type=["jpg", "png", "jpeg", "mp4", "mov"])

if uploaded_file:
    with open(os.path.join("input", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File berhasil diunggah!")

    result_img, result_csv = run_detection(source=os.path.join("input", uploaded_file.name))

    if result_img:
        st.image(result_img, caption="Hasil Deteksi", use_column_width=True)
        st.download_button("📥 Unduh Gambar Deteksi", open(result_img, "rb"), file_name="hasil_deteksi.jpg")

    if result_csv:
        df = pd.read_csv(result_csv)
        st.dataframe(df)
        st.download_button("📥 Unduh CSV Deteksi", open(result_csv, "rb"), file_name="hasil_deteksi.csv")
