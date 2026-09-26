import streamlit as st
import av
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import pyautogui

# Desativa o failsafe do pyautogui para não travar o mouse
pyautogui.FAILSAFE = False 

st.set_page_config(page_title="YOLO + Mouse", layout="centered")
st.title("👁️ Controle de Mouse com YOLO (Sem MediaPipe)")
st.write("Mostre uma **Mãor** para a câmera para mover o mouse 10 pixels para a direita.")

# 1. Carrega o modelo YOLO
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # --- ETAPA 1: YOLO (Detecção de Objetos) ---
    results = model.predict(img, imgsz=320, conf=0.5, verbose=False)
    img_anotada = results[0].plot() # Desenha as caixas
    
    # --- ETAPA 2: Acionar o PyAutoGUI via YOLO ---
    # Inspeciona tudo o que o YOLO encontrou neste frame
    for box in results[0].boxes:
        classe_id = int(box.cls[0])
        nome_classe = model.names[classe_id]
        
        # A classe 'cell phone' é o id 67 no dataset padrão do YOLO
        if nome_classe == "Hand":
            # Aviso visual na tela
            cv2.putText(img_anotada, "Mão Dedectada - Movendo Mouse", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Move o mouse físico
            try:
                pyautogui.move(10, 0)
            except:
                pass
            
            # Interrompe o loop para não mover rápido demais se houver dois celulares
            break 

    return av.VideoFrame.from_ndarray(img_anotada, format="bgr24")

# 3. Inicia o WebRTC
webrtc_streamer(
    key="yolo-mouse-celular",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=process_frame,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)