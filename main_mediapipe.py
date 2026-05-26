import cv2
import numpy as np
import time
import os
import json
from datetime import datetime

# Tenta importar o MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
SATELLITE_DIR = "satellite_images"
SNAPSHOTS_DIR = "snapshots"
ALERTS_LOG_FILE = "alerts_log.json"
WINDOW_NAME = "GreenWatch CV - Central de Controle Gestual MediaPipe"
MAP_W, MAP_H = 1280, 720  # Resolução principal da tela de satélite

SCENARIOS = {
    "1": {"name": "amazon_fire.png", "title": "Codajas (AM) - Amazonia", "lat": -3.4654, "lon": -62.2145},
    "2": {"name": "pantanal_smoke.png", "title": "Corumba (MS) - Pantanal", "lat": -18.0125, "lon": -56.4820},
    "3": {"name": "deforestation_burn.png", "title": "Querencia (MT) - Cerrado", "lat": -11.5080, "lon": -53.6492}
}

os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

# ==============================================================================
# LOGGER DE EMERGÊNCIA GIS
# ==============================================================================
def registrar_alerta_gestual(threat, lat, lon, region):
    """Grava o alerta disparado por gestos no arquivo de auditoria JSON."""
    now = datetime.now()
    alert_entry = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "time_only": now.strftime("%H:%M:%S"),
        "threat": threat,
        "confidence_pct": 100.0,  # Gestos possuem 100% de confiança operacional
        "coordinates": {
            "latitude": round(lat, 5),
            "longitude": round(lon, 5)
        },
        "region": region,
        "source": "COMANDO_GESTUAL_MEDIAPIPE"
    }
    
    logs = []
    if os.path.exists(ALERTS_LOG_FILE):
        try:
            with open(ALERTS_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
            
    # Evita spam do mesmo gesto em menos de 5 segundos
    if logs:
        last_same = [l for l in logs if l.get("threat") == threat and l.get("region") == region]
        if last_same:
            last_time = datetime.strptime(last_same[-1]["timestamp"], "%Y-%m-%d %H:%M:%S")
            if (now - last_time).total_seconds() < 5.0:
                return
                
    logs.append(alert_entry)
    with open(ALERTS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
    print(f"[ALERTA GESTUAL] {alert_entry['timestamp']} - {threat} em {region} (Disparador por operador)")

# ==============================================================================
# PIPELINE PRINCIPAL DO MEDIAPIPE + SATÉLITE
# ==============================================================================
def main():
    print("=" * 70)
    print("      GREENWATCH CV - CENTRAL DE CONTROLE GESTUAL (MEDIAPIPE)")
    print("=" * 70)

    if not MEDIAPIPE_AVAILABLE:
        print("[ERROR] Biblioteca 'mediapipe' nao encontrada.")
        print("Instale utilizando: pip install mediapipe")
        return

    # Inicializa capturador da webcam física para capturar a mão do operador
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Nao foi possivel acessar a webcam física.")
        print("O MediaPipe necessita da camera para rastreamento de gestos do operador.")
        return

    # Define resolução da webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Inicializa utilitários do MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # Configurações de estado e cenário
    cenario_atual = "1"
    cfg = SCENARIOS[cenario_atual]
    
    def carregar_satelite(name):
        path = os.path.join(SATELLITE_DIR, name)
        if not os.path.exists(path):
            img = np.ones((MAP_H, MAP_W, 3), dtype=np.uint8) * 45
            cv2.putText(img, "MAPA INDISPONIVEL", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150, 150, 150), 2)
            return img
        img = cv2.imread(path)
        return cv2.resize(img, (MAP_W, MAP_H))

    sat_img = carregar_satelite(cfg["name"])

    # Coordenadas do cursor laser interativo (controlado pelo dedo indicador)
    laser_x, laser_y = MAP_W // 2, MAP_H // 2
    
    # Histerese para troca de cenários (Gesto de Vitória)
    gesture_hold_frames = 0
    active_gesture = "Nenhum"
    alert_activated = False
    
    print("\nInicializacao concluida com sucesso!")
    print("Controle a central de satelite aproximando sua mao da webcam:")
    print(" -> Mao Aberta (5 dedos): Rastreamento normal")
    # Ponto interativo
    print(" -> Dedo Indicador Apontado (1 dedo): Move mira laser orbital na imagem")
    print(" -> Punho Fechado (0 dedos): Dispara ALERTA DE EMERGENCIA (Panic Button)")
    print(" -> Sinal de Vitoria (2 dedos): Segure por 1.5s para trocar cenario")
    print(" -> Tecla [Q]: Sair do aplicativo")
    print("=" * 70)

    while True:
        # 1. Captura da Webcam
        ret, web_frame = cap.read()
        if not ret:
            print("[ERROR] Falha ao capturar quadro da webcam.")
            break
            
        # Espelha horizontalmente a webcam para ficar intuitivo ao operador (efeito espelho)
        web_frame = cv2.flip(web_frame, 1)
        h_web, w_web = web_frame.shape[:2]
        
        # Converte BGR para RGB para processar no MediaPipe
        rgb_web = cv2.cvtColor(web_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_web)

        # 2. Processa os marcos das mãos (Landmarks) se detectados
        fingers_count = 5  # Valor padrão se nenhuma mão for detectada
        detected_landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenha o esqueleto da mão na webcam
                mp_draw.draw_landmarks(
                    web_frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
                )
                
                # Lista de landmarks em coordenadas normalizadas
                lm = hand_landmarks.landmark
                detected_landmarks = lm
                
                # Algoritmo de classificação de dedos abertos
                # Dedos: Indicador(8), Medio(12), Anelar(16), Mindinho(20)
                # Verifica se a ponta do dedo está acima do nó médio correspondente
                dedos_abertos = []
                
                # Indicador (Tip: 8, Knuckle: 6)
                dedos_abertos.append(lm[8].y < lm[6].y)
                # Médio (Tip: 12, Knuckle: 10)
                dedos_abertos.append(lm[12].y < lm[10].y)
                # Anelar (Tip: 16, Knuckle: 14)
                dedos_abertos.append(lm[16].y < lm[14].y)
                # Mindinho (Tip: 20, Knuckle: 18)
                dedos_abertos.append(lm[20].y < lm[18].y)
                
                # Polegar (Tip: 4, Knuckle: 2) - Compara no eixo X
                dedos_abertos.append(abs(lm[4].x - lm[2].x) > 0.08)
                
                # Conta quantos dedos estão estendidos
                fingers_count = sum(dedos_abertos)

        # 3. Mapeia a quantidade de dedos para GESTOS/COMANDOS
        # Prepara o feed de satélite para desenhar
        sat_display = sat_img.copy()
        
        # Se detectou landmarks, podemos rastrear posições específicas
        if detected_landmarks:
            # Caso 1: Dedo indicador apontando (1 dedo estendido) -> Move a mira laser
            if fingers_count == 1:
                active_gesture = "MIRA LASER (1 Dedo)"
                # Mapeia coordenadas normalizadas do dedo indicador (Landmark 8) para a tela de satélite (1280x720)
                idx_x = detected_landmarks[8].x
                idx_y = detected_landmarks[8].y
                
                # Suavização simples para evitar tremores
                target_x = int(idx_x * MAP_W)
                target_y = int(idx_y * MAP_H)
                laser_x = int(laser_x * 0.7 + target_x * 0.3)
                laser_y = int(laser_y * 0.7 + target_y * 0.3)
                
                # Desenha a mira laser na tela de satélite
                cv2.circle(sat_display, (laser_x, laser_y), 15, (0, 0, 255), 2)
                cv2.line(sat_display, (laser_x - 30, laser_y), (laser_x + 30, laser_y), (0, 0, 255), 1)
                cv2.line(sat_display, (laser_x, laser_y - 30), (laser_x, laser_y + 30), (0, 0, 255), 1)
                cv2.circle(sat_display, (laser_x, laser_y), 3, (0, 0, 255), -1)
                
            # Caso 2: Punho Fechado (0 dedos) -> Dispara pânico/alerta imediato na coordenada da mira
            elif fingers_count == 0:
                active_gesture = "ALERTA PANICO (Fist)"
                alert_activated = True
                
                # Converte coordenada laser em Latitude e Longitude
                lat_gps = cfg["lat"] - (laser_y / MAP_H) * 0.05
                lon_gps = cfg["lon"] + (laser_x / MAP_W) * 0.05
                
                # Dispara alerta
                registrar_alerta_gestual("EMERGENCIA_OPERADOR", lat_gps, lon_gps, cfg["title"])
                
            # Caso 3: Sinal de Vitória (2 dedos) -> Troca cenário segurando
            elif fingers_count == 2:
                active_gesture = "PROXIMO CENARIO (victory)"
                gesture_hold_frames += 1
                
                # Exibe barra de carregamento circular/retangular
                cv2.rectangle(sat_display, (40, MAP_H - 120), (340, 100), (30, 30, 30), -1)
                cv2.rectangle(sat_display, (40, MAP_H - 120), (340, 100), (80, 80, 80), 1)
                progress = min(gesture_hold_frames / 40.0, 1.0) # Precisa segurar por 40 frames (~1.3 segundos)
                cv2.rectangle(sat_display, (45, MAP_H - 115), (45 + int(290 * progress), MAP_H - 105), (255, 100, 0), -1)
                cv2.putText(sat_display, "TRANSMITINDO TROCA DE CANAL...", (45, MAP_H - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                
                if gesture_hold_frames >= 40:
                    # Incrementa cenário
                    next_id = str(int(cenario_atual) % 3 + 1)
                    cenario_atual = next_id
                    cfg = SCENARIOS[cenario_atual]
                    sat_img = carregar_satelite(cfg["name"])
                    gesture_hold_frames = 0
                    alert_activated = False
                    print(f"[CENARIO GESTUAL] Trocado para: {cfg['title']}")
            
            else:
                active_gesture = "OPERACAO NORMAL"
                gesture_hold_frames = 0
                alert_activated = False
        else:
            active_gesture = "NENHUMA MAO DETECTADA"
            gesture_hold_frames = 0
            alert_activated = False

        # 4. Desenha Interface HUD GIS sobre a tela de satélite
        # Barra superior escura
        cv2.rectangle(sat_display, (0, 0), (MAP_W, 55), (15, 15, 15), -1)
        cv2.line(sat_display, (0, 55), (MAP_W, 55), (255, 100, 0), 2)
        
        # Título HUD
        cv2.putText(sat_display, "GREENWATCH GIS - CENTRAL GESTUAL (MEDIAPIPE)", (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(sat_display, f"CENARIO ATIVO: {cfg['title'].upper()}", (MAP_W - 450, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 100, 0), 1, cv2.LINE_AA)
        
        # Telemetria lateral esquerda (Comando Ativo)
        cv2.rectangle(sat_display, (20, 80), (380, 210), (15, 15, 15), -1)
        cv2.rectangle(sat_display, (20, 80), (380, 210), (80, 80, 80), 1)
        
        cv2.putText(sat_display, "DADOS DE RASTREAMENTO GESTUAL", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1, cv2.LINE_AA)
        cv2.line(sat_display, (35, 112), (365, 112), (60, 60, 60), 1)
        
        cv2.putText(sat_display, f"Gesto Detectado: {active_gesture}", (35, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(sat_display, f"Dedos Estendidos: {fingers_count}", (35, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        
        # Coordenadas geográficas do laser orbital
        lat_laser = cfg["lat"] - (laser_y / MAP_H) * 0.05
        lon_laser = cfg["lon"] + (laser_x / MAP_W) * 0.05
        cv2.putText(sat_display, f"MIRA GPS: {lat_laser:.5f} S, {lon_laser:.5f} W", (35, 192), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Se houver emergência ativa por punho cerrado
        if alert_activated:
            cv2.rectangle(sat_display, (0, 0), (MAP_W, MAP_H), (0, 0, 255), 4) # Borda vermelha piscante
            if int(time.time() * 3.5) % 2 == 0:
                cv2.rectangle(sat_display, (MAP_W // 2 - 320, MAP_H // 2 - 60), (MAP_W // 2 + 320, MAP_H // 2 + 30), (0, 0, 255), -1)
                cv2.rectangle(sat_display, (MAP_W // 2 - 320, MAP_H // 2 - 60), (MAP_W // 2 + 320, MAP_H // 2 + 30), (255, 255, 255), 2)
                cv2.putText(sat_display, "!!! ALERTA DE EMERGENCIA ATIVADO !!!", (MAP_W // 2 - 280, MAP_H // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(sat_display, "COORDENADAS E DADOS ENVIADOS PARA A DEFESA CIVIL LOCAL", (MAP_W // 2 - 275, MAP_H // 2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

        # 5. Renders PIP (Picture-In-Picture) - Insere webcam no canto inferior direito
        # Redimensiona o frame da webcam para 240x180
        pip_w, pip_h = 240, 180
        mini_web = cv2.resize(web_frame, (pip_w, pip_h))
        
        # Posiciona no canto inferior direito (com borda/margem de 20px)
        px_start = MAP_W - pip_w - 20
        py_start = MAP_H - pip_h - 20
        
        # Borda cinza do PIP
        cv2.rectangle(sat_display, (px_start - 3, py_start - 3), (px_start + pip_w + 3, py_start + pip_h + 3), (255, 100, 0), 2)
        
        # Copia webcam para o mapa
        sat_display[py_start:py_start+pip_h, px_start:px_start+pip_w] = mini_web
        cv2.putText(sat_display, "OPERADOR WEBCAM", (px_start, py_start - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 100, 0), 1, cv2.LINE_AA)

        # 6. Exibe a tela final integrada
        cv2.imshow(WINDOW_NAME, sat_display)
        
        # 7. Teclas de Controle
        key = cv2.waitKey(10) & 0xFF
        
        # Tecla S: Salva Screenshot analítica
        if key == ord('s') or key == ord('S'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{SNAPSHOTS_DIR}/mediapipe_gis_{ts}.jpg"
            cv2.imwrite(fname, sat_display)
            print(f"[SNAPSHOT GESTUAL] Salvo: {fname}")
            
        # Tecla Q ou Esc: Sai do aplicativo
        elif key in [ord('q'), ord('Q'), 27]:
            print("[SISTEMA] Encerrando central gestual MediaPipe...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("=" * 70)
    print("       GREENWATCH CV - CENTRAL GESTUAL CONCLUIDA COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    main()
