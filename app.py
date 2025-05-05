import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.title("Identificador de PCB TECSCI")

uploaded_file = st.file_uploader("Envie uma imagem para análise", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.button("Testar Modelo"):
        model = YOLO("ultimo.pt")

        file_path = "temp_uploaded_image.jpg"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        results_base = model.predict(source=file_path, conf=0.5, save=True, line_thickness=1)



        save_dir_base = results_base[0].save_dir
        saved_img_path_base = os.path.join(save_dir_base, os.path.basename(file_path))
        img_base = Image.open(saved_img_path_base)
        st.image(img_base, use_container_width=True)

else:
    st.warning("Por favor, envie uma imagem no formato JPG, JPEG ou PNG para continuar.")
