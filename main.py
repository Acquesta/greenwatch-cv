import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
SATELLITE_DIR = "satellite_images"
WINDOW_NAME = "GreenWatch CV - Central de Controle Gestual MediaPipe"
MAP_W, MAP_H = 1280, 720  # Dimensões da tela de satélite do HUD
ZOOM_W, ZOOM_H = 360, 270  # Dimensões da janela de Zoom no HUD
MODEL_PATH = "hand_landmarker.task"

# Conexões anatômicas da mão para desenho do esqueleto
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),      # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12),  # Médio
    (9, 13), (13, 14), (14, 15), (15, 16),# Anelar
    (13, 17), (17, 18), (18, 19), (19, 20),# Mindinho
    (0, 17)                              # Palma
]

SCENARIOS = {
    "1": {
        "name": "amazon_fire.png",
        "title": "Codajas (AM) - Amazonia",
        "lat": -3.4654, "lon": -62.2145,
        "targets": [
            [800, 240, 200, 160, "fogo", 0.94],
            [820, 260, 250, 220, "fumaca", 0.88]
        ]
    },
    "2": {
        "name": "pantanal_smoke.png",
        "title": "Corumba (MS) - Pantanal",
        "lat": -18.0125, "lon": -56.4820,
        "targets": [
            [640, 360, 420, 360, "fumaca", 0.91]
        ]
    },
    "3": {
        "name": "deforestation_burn.png",
        "title": "Querencia (MT) - Cerrado",
        "lat": -11.5080, "lon": -53.6492,
        "targets": [
            [380, 430, 150, 120, "fogo", 0.89],
            [900, 280, 220, 180, "fumaca", 0.82],
            [410, 440, 210, 170, "fumaca", 0.84]
        ]
    }
}

# ==============================================================================
# 2. FUNÇÕES AUXILIARES DE RASTREAMENTO E RENDERIZAÇÃO
# ==============================================================================
def carregar_satelite(scenario_key):
    """Carrega a imagem de satélite do cenário ativo e a redimensiona para o HUD."""
    cfg = SCENARIOS[scenario_key]
    path = os.path.join(SATELLITE_DIR, cfg["name"])
    if not os.path.exists(path):
        img = np.ones((MAP_H, MAP_W, 3), dtype=np.uint8) * 45
        cv2.putText(img, "MAPA INDISPONIVEL", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150, 150, 150), 2)
        return img
    return cv2.resize(cv2.imread(path), (MAP_W, MAP_H))

def desenhar_esqueleto_mao(frame, landmarks, w_frame, h_frame):
    """Desenha as articulações e conexões da mão no quadro da webcam (PIP)."""
    # Desenha as linhas de conexão
    for start, end in HAND_CONNECTIONS:
        pt_start = landmarks[start]
        pt_end = landmarks[end]
        x1, y1 = int(pt_start.x * w_frame), int(pt_start.y * h_frame)
        x2, y2 = int(pt_end.x * w_frame), int(pt_end.y * h_frame)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
    # Desenha os círculos das articulações
    for idx, pt in enumerate(landmarks):
        x, y = int(pt.x * w_frame), int(pt.y * h_frame)
        color = (0, 255, 0) if idx in [4, 8, 12, 16, 20] else (0, 255, 255)
        cv2.circle(frame, (x, y), 4, color, -1)

def contar_dedos_abertos(landmarks):
    """Retorna o número de dedos abertos com base nas posições dos landmarks."""
    dedos = [
        landmarks[8].y < landmarks[6].y,   # Indicador
        landmarks[12].y < landmarks[10].y, # Médio
        landmarks[16].y < landmarks[14].y, # Anelar
        landmarks[20].y < landmarks[18].y, # Mindinho
        abs(landmarks[4].x - landmarks[2].x) > 0.08 # Polegar (limiar lateral)
    ]
    return sum(dedos)

# ==============================================================================
# 3. DETECTOR DE INCÊNDIOS HÍBRIDO (YOLOv8 + Mock)
# ==============================================================================
class ZoomYoloDetector:
    """Carrega o modelo YOLOv8 ou simula detecções táticas caso ausente."""
    def __init__(self):
        self.model = None
        self.yolo_loaded = False
        self.weight_name = "Nenhum"
        try:
            self.model = YOLO("best.pt")
            self.yolo_loaded = True
            self.weight_name = "best.pt"
            print("[YOLO] Modelo 'best.pt' carregado com sucesso!")
        except Exception:
            print("[YOLO] 'best.pt' nao encontrado. Rodando em modo de simulacao.")

    def detect(self, zoom_frame, scenario_key, x1_lupa, y1_lupa, lupa_w, lupa_h):
        """Identifica focos de fogo/fumaça no quadro de zoom ampliado."""
        # Caminho 1: Inferência em tempo real via YOLOv8
        if self.yolo_loaded:
            try:
                detections = []
                results = self.model.predict(source=zoom_frame, conf=0.25, verbose=False)
                for box in results[0].boxes:
                    zx1, zy1, zx2, zy2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = self.model.names[int(box.cls[0])].lower()
                    norm_cls = "fogo" if any(x in cls for x in ["fire", "fogo", "flame"]) else "fumaca"
                    detections.append({
                        "class": norm_cls,
                        "bbox_zoom": (zx1, zy1, zx2 - zx1, zy2 - zy1),
                        "confidence": conf
                    })
                return detections
            except Exception:
                pass

        # Caminho 2: Detector Mock Georreferenciado Proporcional (Fallback)
        detections = []
        cfg = SCENARIOS[scenario_key]
        for tgt in cfg["targets"]:
            tx_center, ty_center, tw, th, cls_name, base_conf = tgt
            tx1, ty1 = tx_center - tw // 2, ty_center - th // 2
            tx2, ty2 = tx_center + tw // 2, ty_center + th // 2
            
            # Interseção da área do foco com a lente de lupa
            ix1, iy1 = max(x1_lupa, tx1), max(y1_lupa, ty1)
            ix2, iy2 = min(x1_lupa + lupa_w, tx2), min(y1_lupa + lupa_h, ty2)
            
            if ix2 > ix1 and iy2 > iy1:
                # Escala proporcionalmente para a janela de Zoom do HUD (360x270)
                zx1 = int((ix1 - x1_lupa) * (ZOOM_W / lupa_w))
                zy1 = int((iy1 - y1_lupa) * (ZOOM_H / lupa_h))
                zx2 = int((ix2 - x1_lupa) * (ZOOM_W / lupa_w))
                zy2 = int((iy2 - y1_lupa) * (ZOOM_H / lupa_h))
                
                detections.append({
                    "class": cls_name,
                    "bbox_zoom": (zx1, zy1, zx2 - zx1, zy2 - zy1),
                    "confidence": base_conf
                })
        return detections

# ==============================================================================
# 4. PIPELINE PRINCIPAL DO SISTEMA
# ==============================================================================
def main():


    if not os.path.exists(MODEL_PATH):
        print("[MEDIAPIPE] Baixando automaticamente 'hand_landmarker.task'...")
        import urllib.request
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", MODEL_PATH)

    # Inicialização da Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam fisica indisponivel.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Configuração do rastreador gestual do MediaPipe Tasks
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    # Inicialização do detector YOLOv8
    yolo_detector = ZoomYoloDetector()

    # Estado Inicial
    cenario_atual = "1"
    cfg = SCENARIOS[cenario_atual]
    sat_img = carregar_satelite(cenario_atual)

    # Coordenadas e dinâmica de Zoom da Lente Móvel
    laser_x, laser_y = MAP_W // 2, MAP_H // 2
    current_lupa_w, current_lupa_h = 160, 120
    
    gesture_hold_frames = 0
    active_gesture = "Nenhum"
    last_timestamp = 0
    fire_first_seen_time = None

    while True:
        ret, web_frame = cap.read()
        if not ret:
            break
            
        web_frame = cv2.flip(web_frame, 1)
        h_web, w_web = web_frame.shape[:2]
        
        # Converte para mp.Image e envia para processamento
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(web_frame, cv2.COLOR_BGR2RGB))
        
        # Cooldown para garantir timestamp estritamente crescente
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp:
            timestamp_ms = last_timestamp + 1
        last_timestamp = timestamp_ms
        
        results = detector.detect_for_video(mp_image, timestamp_ms)
        has_hand = bool(results.hand_landmarks)

        fingers_count = 5
        detected_landmarks = []
        
        # Processa e desenha a mão se detectada
        if has_hand:
            hand_landmarks = results.hand_landmarks[0]
            detected_landmarks = hand_landmarks
            desenhar_esqueleto_mao(web_frame, hand_landmarks, w_web, h_web)
            fingers_count = contar_dedos_abertos(hand_landmarks)

        # Prepara a tela final do HUD e o painel de Zoom
        sat_display = sat_img.copy()
        zoom_display = np.ones((ZOOM_H, ZOOM_W, 3), dtype=np.uint8) * 20
        cv2.putText(zoom_display, "AGUARDANDO SINAL DE LENTE", (40, ZOOM_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 120), 1, cv2.LINE_AA)
        
        lupa_ativa = False
        tem_fogo = False
        
        # Executa as lógicas de gestos e Lupa Inteligente
        if detected_landmarks:
            lupa_ativa = True
            
            # Rastreamento tático baseado no centro da palma (Landmark 9)
            laser_x = int(laser_x * 0.70 + (detected_landmarks[9].x * MAP_W) * 0.30)
            laser_y = int(laser_y * 0.70 + (detected_landmarks[9].y * MAP_H) * 0.30)
            
            # Escolhe o nível de zoom dinâmico com base nos gestos
            if fingers_count == 5:
                target_w, target_h = 96, 72  # Zoom In Máximo (3.75x)
                active_gesture = "ZOOM IN MÁXIMO (Mao Aberta)"
                gesture_hold_frames = 0
            elif fingers_count == 0:
                target_w, target_h = 240, 180  # Zoom Out (1.5x)
                active_gesture = "ZOOM OUT (Mao Fechada)"
                gesture_hold_frames = 0
            elif fingers_count == 2:
                target_w, target_h = 160, 120  # Zoom Médio Padrão
                active_gesture = "PROXIMO CENARIO (victory)"
                gesture_hold_frames += 1
                
                # Barra compacta de progresso para trocar de cenário gestualmente
                cv2.rectangle(sat_display, (40, MAP_H - 120), (340, MAP_H - 75), (30, 30, 30), -1)
                cv2.rectangle(sat_display, (40, MAP_H - 120), (340, MAP_H - 75), (80, 80, 80), 1)
                progress = min(gesture_hold_frames / 40.0, 1.0)
                cv2.rectangle(sat_display, (45, MAP_H - 115), (45 + int(290 * progress), MAP_H - 105), (255, 100, 0), -1)
                cv2.putText(sat_display, "TRANSMITINDO TROCA DE CANAL...", (45, MAP_H - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                
                if gesture_hold_frames >= 40:
                    cenario_atual = str(int(cenario_atual) % 3 + 1)
                    cfg = SCENARIOS[cenario_atual]
                    sat_img = carregar_satelite(cenario_atual)
                    gesture_hold_frames = 0
            else:
                target_w, target_h = 160, 120  # Zoom Médio padrão
                active_gesture = "TRACKING ORBITAL (Zoom Medio)"
                gesture_hold_frames = 0
                
            # Interpolação suave para simular o comportamento de lente física mecânica
            current_lupa_w = int(current_lupa_w * 0.82 + target_w * 0.18)
            current_lupa_h = int(current_lupa_h * 0.82 + target_h * 0.18)
        else:
            active_gesture = "NENHUMA MAO DETECTADA"
            gesture_hold_frames = 0
            # Retorna de forma fluida para as dimensões padrão
            current_lupa_w = int(current_lupa_w * 0.85 + 160 * 0.15)
            current_lupa_h = int(current_lupa_h * 0.85 + 120 * 0.15)

        # Limita e calcula a caixa da lupa no mapa
        x1_lupa = max(0, min(laser_x - current_lupa_w // 2, MAP_W - current_lupa_w))
        y1_lupa = max(0, min(laser_y - current_lupa_h // 2, MAP_H - current_lupa_h))
        x2_lupa, y2_lupa = x1_lupa + current_lupa_w, y1_lupa + current_lupa_h
        
        # Realiza o Crop e Zoom do frame selecionado
        if lupa_ativa or (current_lupa_w != 160 or current_lupa_h != 120):
            cv2.rectangle(sat_display, (x1_lupa, y1_lupa), (x2_lupa, y2_lupa), (0, 255, 0), 2)
            cv2.drawMarker(sat_display, (laser_x, laser_y), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
            
            crop = sat_img[y1_lupa:y2_lupa, x1_lupa:x2_lupa]
            if crop.size > 0:
                zoom_display = cv2.resize(crop, (ZOOM_W, ZOOM_H))
                zoom_detections = yolo_detector.detect(zoom_display, cenario_atual, x1_lupa, y1_lupa, current_lupa_w, current_lupa_h)
                
                tem_fogo = any(d["class"] == "fogo" for d in zoom_detections)
                
                # Desenha marcações de incêndios na janela de Zoom
                for det in zoom_detections:
                    zx, zy, zw, zh = det["bbox_zoom"]
                    color = (0, 0, 255) if det["class"] == "fogo" else (0, 200, 255)
                    cv2.rectangle(zoom_display, (zx, zy), (zx + zw, zy + zh), color, 2)
                    cv2.putText(zoom_display, f"IA: {det['class'].upper()} {det['confidence']*100:.0f}%", 
                                (zx, zy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        # Lógica de cálculo de tempo contínuo de fogo
        if tem_fogo:
            if fire_first_seen_time is None:
                fire_first_seen_time = time.time()
            fire_duration = time.time() - fire_first_seen_time
        else:
            fire_first_seen_time = None
            fire_duration = 0.0

        # RENDERIZAÇÃO DA INTERFACE HUD GIS
        # Barra de Status Superior
        cv2.rectangle(sat_display, (0, 0), (MAP_W, 55), (15, 15, 15), -1)
        cv2.line(sat_display, (0, 55), (MAP_W, 55), (255, 100, 0), 2)
        cv2.putText(sat_display, "GREENWATCH GIS - CENTRAL GESTUAL (MEDIAPIPE)", (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(sat_display, f"CENARIO: {cfg['title'].upper()}", (MAP_W - 450, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 100, 0), 1, cv2.LINE_AA)
        
        # Painel Lateral de Telemetria
        cv2.rectangle(sat_display, (20, 80), (380, 210), (15, 15, 15), -1)
        cv2.rectangle(sat_display, (20, 80), (380, 210), (80, 80, 80), 1)
        cv2.putText(sat_display, "DADOS DE RASTREAMENTO GESTUAL", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1, cv2.LINE_AA)
        cv2.line(sat_display, (35, 112), (365, 112), (60, 60, 60), 1)
        cv2.putText(sat_display, f"Gesto: {active_gesture}", (35, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(sat_display, f"Dedos: {fingers_count}", (35, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        
        zoom_nivel = ZOOM_W / current_lupa_w
        cv2.putText(sat_display, f"ZOOM ATIVO: {zoom_nivel:.2f}x", (35, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
        
        lat_laser = cfg["lat"] - (laser_y / MAP_H) * 0.05
        lon_laser = cfg["lon"] + (laser_x / MAP_W) * 0.05
        cv2.putText(sat_display, f"LENTE GPS: {lat_laser:.5f} S, {lon_laser:.5f} W", (35, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Painel de Análise de Zoom de IA
        px_zoom, py_zoom = 20, 230
        cv2.rectangle(sat_display, (px_zoom, py_zoom), (px_zoom + ZOOM_W, py_zoom + ZOOM_H), (15, 15, 15), -1)
        cv2.rectangle(sat_display, (px_zoom - 3, py_zoom - 3), (px_zoom + ZOOM_W + 3, py_zoom + ZOOM_H + 3), (255, 100, 0), 2)
        sat_display[py_zoom:py_zoom+ZOOM_H, px_zoom:px_zoom+ZOOM_W] = zoom_display
        
        cv2.rectangle(sat_display, (px_zoom, py_zoom), (px_zoom + 180, py_zoom + 22), (15, 15, 15), -1)
        cv2.putText(sat_display, f"ZOOM IA ({yolo_detector.weight_name})", (px_zoom + 6, py_zoom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        # Efeito de Alerta Vermelho Piscante (Fogo Confirmado por 5s)
        if fire_first_seen_time is not None and fire_duration >= 5.0:
            if int(time.time() * 3.3) % 2 == 0:
                cv2.rectangle(sat_display, (0, 0), (MAP_W, MAP_H), (0, 0, 255), 6)
                bx, by, bw, bh = MAP_W // 2 - 280, 80, 560, 50
                cv2.rectangle(sat_display, (bx, by), (bx + bw, by + bh), (0, 0, 255), -1)
                cv2.rectangle(sat_display, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
                cv2.putText(sat_display, "!!! INCENDIO CONFIRMADO - ALERTA VERMELHO !!!", (bx + 25, by + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        # Câmera do Operador PIP (Webcam) no canto inferior direito
        pip_w, pip_h = 240, 180
        px_start, py_start = MAP_W - pip_w - 20, MAP_H - pip_h - 20
        cv2.rectangle(sat_display, (px_start - 3, py_start - 3), (px_start + pip_w + 3, py_start + pip_h + 3), (255, 100, 0), 2)
        sat_display[py_start:py_start+pip_h, px_start:px_start+pip_w] = cv2.resize(web_frame, (pip_w, pip_h))
        cv2.putText(sat_display, "OPERADOR WEBCAM", (px_start, py_start - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 100, 0), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, sat_display)
        
        key = cv2.waitKey(10) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break

    cap.release()
    try:
        detector.close()
    except Exception:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
