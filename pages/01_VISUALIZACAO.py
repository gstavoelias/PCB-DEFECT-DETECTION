import streamlit as st
from pymongo import MongoClient
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    layout="wide",
    page_title="Dashboard Inspeção Visual"
)
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)


client = MongoClient("localhost:27017")
db = client.get_database("tecsci")
pcbs = db.get_collection("pcb").find()
df = pd.json_normalize(pcbs).drop(["_id", "annotations", "img_path"], axis=1)
df["horario"] = pd.to_datetime(df["horario"])

st.header("Dashboard - Inspeções da IA")
col1, col2, col3, col4 = st.columns(4, gap="small")
with col1:
    with st.container(border=True):
        st.markdown(f'<p class="btc_text">Total de Erros<br></p><p class="price_details">{df["missing"].sum() + df["reversed"].sum() + df["false_negatives"].sum() - df["false_positives"].sum()}</p>', unsafe_allow_html = True)
with col2:
    with st.container(border=True):
        st.markdown(f'<p class="eth_text">TCUs com Erro<br></p><p class="price_details">{100 - 100*len(df[(df['missing'] == 0) & (df['reversed'] == 0) & (df['false_negatives'] == 0)])/len(df)}%</p>', unsafe_allow_html = True)
with col3:
    with st.container(border=True):
        st.markdown(f'<p class="sol_text ">Precisão<br></p><p class="price_details">{round(100*(1 - df["false_positives"].sum()/(df["missing"].sum() + df["reversed"].sum())), 2)}%</p>', unsafe_allow_html = True)
with col4:
    with st.container(border=True):
        st.markdown(f'<p class="xrp_text ">Sensibilidade<br></p><p class="price_details">{round(100*(1 - df["false_negatives"].sum()/(df["missing"].sum() + df["reversed"].sum())), 2)}%</p>', unsafe_allow_html = True)

col5, col6 = st.columns(2, gap="small")

with col5:
    with st.container(border=True):
        data = df.groupby(df["horario"].dt.date).sum(numeric_only=True)
        fig = go.Figure()
        fig.add_bar(x=data.index, y=data["missing"] - data["false_positives"], name="Componentes Faltantes", marker_color=px.colors.sequential.Inferno[0])
        fig.add_bar(x=data.index, y=data["reversed"], name="Componentes Invertidos", marker_color=px.colors.sequential.Inferno[1])
        fig.update_layout(title={'text': 'Erros por dia'})
        st.plotly_chart(fig, use_container_width=True)
                

with col6:
    with st.container(border=True):
        fig2 = px.pie(names=["Infinity", "Enterplak"], values=[70, 30], color_discrete_sequence= px.colors.sequential.Inferno, title="Erros por Empresa")
        st.plotly_chart(fig2, use_container_width=True)

with st.expander("Base de Dados"):
    st.dataframe(df)