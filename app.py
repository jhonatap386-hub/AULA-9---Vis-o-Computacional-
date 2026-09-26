import streamlit as st
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="Visão Computacional com YOLO", layout="centered")
st.title("👁️ Detecção em Tempo Real com YOLOv8 e Streamlit")
st.write("Esta aplicação utiliza inteligência artificial leve (YOLOv8 Nano) para identificar pessoas e objetos ao vivo através da sua webcam.")

# Carrega o modelo YOLO (o arquivo .pt será baixado automaticamente na primeira execução)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Função que processa cada frame de vídeo capturado pelo navegador
def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Realiza a inferência com o YOLO (resolução 320 para garantir fluidez)
    results = model.predict(img, imgsz=320, conf=0.5, verbose=False)
    
    # Desenha as caixas delimitadoras e os nomes das classes na imagem
    img_anotada = results[0].plot()
    
    return av.VideoFrame.from_ndarray(img_anotada, format="bgr24")

# Inicializa o componente de streaming WebRTC para capturar a webcam
webrtc_streamer(
    key="yolo-webcam",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_frame,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)