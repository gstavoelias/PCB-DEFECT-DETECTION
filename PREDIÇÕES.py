import streamlit as st
from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from pymongo import MongoClient
import cv2
from datetime import datetime

st.set_page_config(
    layout="centered",
    page_title="Inspeção Visual - TECSCI"
)

client = MongoClient("localhost:27017")
db = client.get_database("tecsci")
pcbs = db.get_collection("pcb")

capacitor_list = [
    (2439.918212890625, 2556.60693359375, 834.887939453125, 953.5704956054688, "Capacitor_D"),
    (1321.389404296875, 1464.6134033203125, 282.0599670410156, 419.3526916503906, "Capacitor_D"),
    (966.6650390625, 1153.9564208984375, 489.9255676269531, 643.5599365234375, "Capacitor_R"),
    (2557.781005859375, 2708.772216796875, 1721.9512939453125, 1891.53955078125, "Capacitor_D"),
    (1035.885498046875, 1191.3262939453125, 287.7256774902344, 420.1536560058594, "Capacitor_R"),
    (2172.884765625, 2312.7177734375, 1132.0899658203125, 1288.695068359375, "Capacitor_U"),
    (1797.166015625, 1938.33642578125, 1868.6573486328125, 2037.5247802734375, "Capacitor_L"),
    (1940.3585205078125, 2086.7919921875, 1335.6978759765625, 1485.851318359375, "Capacitor_L"),
]

st.title("Inspeção Visual por IA")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

with st.expander(label="Upload da Imagem", expanded=False):
    st.session_state["uploaded_file"] = st.file_uploader("Envie uma imagem para análise", type=["jpg", "jpeg", "png"], label_visibility="collapsed")



def draw_image(image, cls_name, color, x1, x2, y1, y2):
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)


if st.session_state.uploaded_file is not None:
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="my_model.pt",
        confidence_threshold=0.3,
    )

    file_path = "temp_uploaded_image.jpg"
    with open(file_path, "wb") as f:
        f.write(st.session_state["uploaded_file"].getbuffer())
    result = get_prediction(file_path, model)
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictions = result.object_prediction_list  
    missing_count = 0
    reversed_count = 0
    for prediction in predictions:
        if prediction.category.id == 6:
            draw_image(img, prediction.category.name,(0, 255, 0), prediction.bbox.minx, prediction.bbox.maxx, prediction.bbox.miny, prediction.bbox.maxy)
            missing_count += 1
        if prediction.category.id == 1 or prediction.category.id == 9:
            draw_image(img, prediction.category.name, (255, 0, 0), prediction.bbox.minx, prediction.bbox.maxx, prediction.bbox.miny, prediction.bbox.maxy)
            reversed_count += 1
        if prediction.category.id in (2, 3, 4, 5):
            center_x = prediction.bbox.minx + (prediction.bbox.maxx-prediction.bbox.minx)/2
            center_y = prediction.bbox.miny + (prediction.bbox.maxy - prediction.bbox.miny)/2
            for capacitor in capacitor_list:
                if capacitor[0] < center_x < capacitor[1] and capacitor[2] < center_y < capacitor[3]:
                    if capacitor[4] != prediction.category.name:
                        draw_image(img, prediction.category.name, (255, 0, 0), prediction.bbox.minx, prediction.bbox.maxx, prediction.bbox.miny, prediction.bbox.maxy)
                        reversed_count += 1
    col1, col2 = st.columns(2, gap='small')
    with col1:
        with st.container(border=True):
            st.markdown(f'<p class="btc_text">Componentes Faltando<br></p><p class="price_details">{missing_count}</p>', unsafe_allow_html = True)
    with col2:
        with st.container(border=True):
            st.markdown(f'<p class="btc_text">Componentes Invertidos<br></p><p class="price_details">{reversed_count}</p>', unsafe_allow_html = True)

    st.image(img,width=4608)

    st.subheader("Envio do Relatório")
    with st.form("my_form"):
        false_positives = st.number_input("Falsos Positivos:", min_value=0, max_value=100)
        false_negatives = st.number_input("Problemas não identificados:",  min_value=0, max_value=100)
        observacao = st.text_area("Comentário (Opcional)")
        enviado = st.form_submit_button("Enviar", on_click=None, type="primary")
        if enviado:
            data = {
                "annotations": result.to_coco_annotations(),
                "img_path": None,
                "missing": int(missing_count),
                "reversed": int(reversed_count),
                "false_positives": int(false_positives),
                "false_negatives":  int(false_negatives),
                "horario": datetime.now().isoformat(),
                "obs": observacao
            }
            response = pcbs.insert_one(data)
else:
    st.warning("Por favor, envie uma imagem no formato JPG, JPEG ou PNG para continuar.")





