import cv2
import numpy as np
import time
import os
import json
import math
from datetime import datetime

# Tenta importar a biblioteca ultralytics para suporte a YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
SNAPSHOTS_DIR = "snapshots"
ALERTS_LOG_FILE = "alerts_log.json"
WINDOW_NAME = "GreenWatch CV - Sistema de Sensoriamento Orbital e Alertas"
VIEW_W, VIEW_H = 640, 640 # Resolução nativa de processamento e entrada do YOLO
HUD_W, HUD_H = 1280, 720  # Resolução da tela final do Dashboard

SATELLITE_DIR = "satellite_images"
SCENARIOS = {
    "1": {
        "name": "amazon_fire.png",
        "title": "Floresta Amazonica (Focos Ativos)",
        "lat": -3.4654,
        "lon": -62.2145,
        "region": "Codajas - AM",
        # Focos absolutos na escala 1600x1200 [x_center, y_center, width, height, class]
        "targets": [
            [1000, 380, 240, 200, "fogo", 0.94],
            [1020, 420, 320, 280, "fumaca", 0.88]
        ]
    },
    "2": {
        "name": "pantanal_smoke.png",
        "title": "Bacia do Pantanal (Pluma de Fumaca)",
        "lat": -18.0125,
        "lon": -56.4820,
        "region": "Corumba - MS",
        "targets": [
            [800, 600, 520, 450, "fumaca", 0.91]
        ]
    },
    "3": {
        "name": "deforestation_burn.png",
        "title": "Cerrado / Amazonia (Cicatrizes de Queimada)",
        "lat": -11.5080,
        "lon": -53.6492,
        "region": "Querencia - MT",
        "targets": [
            [480, 720, 180, 150, "fogo", 0.89],
            [1120, 460, 280, 220, "fumaca", 0.82],
            [510, 730, 260, 210, "fumaca", 0.84]
        ]
    }
}

# Inicializa pastas e arquivos
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(SATELLITE_DIR, exist_ok=True)
if not os.path.exists(ALERTS_LOG_FILE):
    with open(ALERTS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# ==============================================================================
# REGISTRO DE ALERTAS GIS
# ==============================================================================
class GISAlertLogger:
    @staticmethod
    def log_alert(threat, confidence, lat, lon, area_ha, region):
        """Grava alerta no JSON com metadados georreferenciados."""
        now = datetime.now()
        alert_entry = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "time_only": now.strftime("%H:%M:%S"),
            "threat": threat.upper(),
            "confidence_pct": round(confidence * 100, 1),
            "coordinates": {
                "latitude": round(lat, 5),
                "longitude": round(lon, 5)
            },
            "area_hectares": round(area_ha, 1),
            "region": region,
            "satellite": "GREENWATCH-1"
        }
        
        try:
            with open(ALERTS_LOG_FILE, "r", encoding="utf-8") as file:
                logs = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            logs = []
            
        # Evita spam: registra apenas se passaram 5 segundos do último alerta idêntico na mesma região
        should_write = True
        if logs:
            last_same = [l for l in logs if l.get("threat") == alert_entry["threat"] and l.get("region") == region]
            if last_same:
                last_time_str = last_same[-1].get("timestamp")
                if last_time_str:
                    last_time = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
                    if (now - last_time).total_seconds() < 5.0:
                        should_write = False
                    
        if should_write:
            logs.append(alert_entry)
            with open(ALERTS_LOG_FILE, "w", encoding="utf-8") as file:
                json.dump(logs, file, ensure_ascii=False, indent=4)
            print(f"[ALERTA ORBITAL] {alert_entry['timestamp']} - {alert_entry['threat']} em {region} (Área: {area_ha} ha)")
            
        return alert_entry

# ==============================================================================
# MOTOR DETECTOR DE IA (YOLOv8 REAL + MOCK GEORREFERENCIADO)
# ==============================================================================
class SatelliteYoloDetector:
    def __init__(self):
        self.model = None
        self.yolo_loaded = False
        self.weight_name = "Nenhum"
        self.init_model()

    def init_model(self):
        if not YOLO_AVAILABLE:
            print("[YOLO] Biblioteca 'ultralytics' não encontrada. Rodando no modo de simulação georreferenciada.")
            return

        # Busca pesos para carregar
        weights_options = ["fire_smoke_yolo.pt", "best.pt", "yolov8s.pt", "yolov8n.pt"]
        for weight in weights_options:
            try:
                print(f"[YOLO] Tentando carregar pesos: '{weight}'...")
                self.model = YOLO(weight)
                self.yolo_loaded = True
                self.weight_name = weight
                print(f"[YOLO] IA carregada com sucesso com pesos: '{weight}'!")
                break
            except Exception as e:
                print(f"[YOLO] Não foi possível carregar '{weight}': {str(e)}")

    def detect(self, viewport_frame, scenario_key, sweep_x, sweep_y):
        """
        Executa detecção sobre o recorte de satélite ativo.
        Caso o arquivo de pesos real não esteja na pasta, ele executa um detector MOCK
        baseado na correspondência exata de coordenadas do satélite na imagem de alta resolução.
        Isso garante o funcionamento imediato do pipeline orbital e dos logs.
        """
        if self.yolo_loaded:
            # Inferência Real do YOLOv8
            try:
                results = self.model.predict(source=viewport_frame, conf=0.25, verbose=False)
                detections = []
                
                for box in results[0].boxes:
                    coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, coords)
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    class_name = self.model.names[cls_id].lower()
                    
                    # Normaliza classes de fogo/fumaça
                    normalized_class = ""
                    if "fire" in class_name or "fogo" in class_name or "flame" in class_name:
                        normalized_class = "fogo"
                    elif "smoke" in class_name or "fumaca" in class_name or "gray" in class_name:
                        normalized_class = "fumaca"
                    else:
                        normalized_class = class_name
                        
                    detections.append({
                        "class": normalized_class,
                        "bbox": (x1, y1, w, h),
                        "confidence": conf,
                        "center": (x1 + w//2, y1 + h//2)
                    })
                return detections
            except Exception as e:
                print(f"[YOLO ERROR] Falha na inferência real: {e}. Alternando para simulação orbital.")
        
        # Detector MOCK Georreferenciado
        # Projetar caixas absolutas do cenário para o recorte local de 640x640 da varredura
        detections = []
        cfg = SCENARIOS[scenario_key]
        
        for tgt in cfg["targets"]:
            tx_center, ty_center, tw, th, cls_name, base_conf = tgt
            
            # Limites absolutos do alvo
            tx1 = tx_center - tw // 2
            ty1 = ty_center - th // 2
            tx2 = tx_center + tw // 2
            ty2 = ty_center + th // 2
            
            # Verifica interseção com a janela móvel (sweep_x, sweep_y) a (sweep_x+640, sweep_y+640)
            ix1 = max(sweep_x, tx1)
            iy1 = max(sweep_y, ty1)
            ix2 = min(sweep_x + VIEW_W, tx2)
            iy2 = min(sweep_y + VIEW_H, ty2)
            
            # Área de sobreposição
            inter_w = ix2 - ix1
            inter_h = iy2 - iy1
            
            if inter_w > 40 and inter_h > 40:
                # Coordenadas relativas à janela móvel de 640x640
                rx = int(ix1 - sweep_x)
                ry = int(iy1 - sweep_y)
                rw = int(inter_w)
                rh = int(inter_h)
                
                detections.append({
                    "class": cls_name,
                    "bbox": (rx, ry, rw, rh),
                    "confidence": base_conf + np.random.uniform(-0.02, 0.02),
                    "center": (rx + rw//2, ry + rh//2)
                })
                
        return detections

# ==============================================================================
# DASHBOARD HUD GIS
# ==============================================================================
class GISDashboardHUD:
    def __init__(self):
        self.alert_history = []
        self.scan_line_y = 0
        
    def add_alert(self, alert):
        if not self.alert_history or self.alert_history[0]["timestamp"] != alert["timestamp"] or self.alert_history[0]["threat"] != alert["threat"]:
            self.alert_history.insert(0, alert)
            if len(self.alert_history) > 5:
                self.alert_history.pop()

    def draw(self, viewport_frame, detections, active_scenario, sweep_x, sweep_y, img_w, img_h):
        # 1. Cria a base preta do HUD (1280x720)
        hud = np.zeros((HUD_H, HUD_W, 3), dtype=np.uint8)
        
        # Copia o feed recortado do satélite (640x640) para o centro esquerdo do HUD
        # Posicionado de y=40 a y=680, e x=40 a x=680
        hud[40:680, 40:680] = viewport_frame
        
        # 2. Desenha o Mini-Mapa de Varredura (Posição orbital no canto direito superior)
        # Redimensiona a imagem orbital inteira para caber no mini-mapa (240x180)
        map_w, map_h = 240, 180
        map_x, map_y = HUD_W - 280, 100
        
        # Cria retângulo cinza de fundo para o mini-mapa
        cv2.rectangle(hud, (map_x - 5, map_y - 5), (map_x + map_w + 5, map_y + map_h + 5), (50, 50, 50), -1)
        
        # Carrega e escala a imagem original do cenário
        sc_name = active_scenario["name"]
        full_img_path = os.path.join(SATELLITE_DIR, sc_name)
        if os.path.exists(full_img_path):
            full_img = cv2.imread(full_img_path)
            if full_img is not None:
                mini_map = cv2.resize(full_img, (map_w, map_h))
                hud[map_y:map_y+map_h, map_x:map_x+map_w] = mini_map
            
            # Desenha retângulo verde indicando a janela de varredura ativa
            rx = int((sweep_x / img_w) * map_w)
            ry = int((sweep_y / img_h) * map_h)
            rw = int((VIEW_W / img_w) * map_w)
            rh = int((VIEW_H / img_h) * map_h)
            cv2.rectangle(hud, (map_x + rx, map_y + ry), (map_x + rx + rw, map_y + ry + rh), (0, 255, 0), 2)
            cv2.putText(hud, "TRACKING ORBITAL", (map_x, map_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)
            
        # 3. Informações Orbitais (Canto Direito)
        info_x = HUD_W - 280
        cv2.putText(hud, "TELEMETRIA DE SATELITE", (info_x, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 205, 50), 1, cv2.LINE_AA)
        cv2.line(hud, (info_x, 323), (HUD_W - 40, 323), (80, 80, 80), 1)
        
        cv2.putText(hud, "Satelite: GREENWATCH-1", (info_x, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Orbita: Polar Heliossincrona", (info_x, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Sensor: Multispectral L1T", (info_x, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Regiao: {active_scenario['region']}", (info_x, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
        
        # Coordenadas geográficas estimadas do centro da câmera
        center_lat = active_scenario["lat"] - (sweep_y / img_h) * 0.05
        center_lon = active_scenario["lon"] + (sweep_x / img_w) * 0.05
        cv2.putText(hud, f"Lat: {center_lat:.5f} S", (info_x, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Lon: {center_lon:.5f} W", (info_x, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Histórico de Alertas Orbitais no HUD
        cv2.putText(hud, "LOG DE RISCOS DETECTADOS", (info_x, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(hud, (info_x, 498), (HUD_W - 40, 498), (80, 80, 80), 1)
        
        if not self.alert_history:
            cv2.putText(hud, "Nenhuma anomalia ativa.", (info_x, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)
        else:
            for idx, al in enumerate(self.alert_history[:3]):
                ly = 525 + (idx * 32)
                color = (0, 0, 255) if al["threat"] == "FOGO" else (0, 200, 255)
                # Sinalizador colorido
                cv2.rectangle(hud, (info_x, ly - 9), (info_x + 5, ly), color, -1)
                
                # Texto georreferenciado
                txt = f"[{al['time_only']}] {al['threat']} ({al['area_hectares']} ha)"
                cv2.putText(hud, txt, (info_x + 15, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(hud, f"Coord: {al['coordinates']['latitude']:.4f}S / {al['coordinates']['longitude']:.4f}W", (info_x + 15, ly + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)

        # 5. Barra Superior de Status do Sistema
        cv2.rectangle(hud, (0, 0), (HUD_W, 40), (25, 25, 25), -1)
        cv2.line(hud, (0, 40), (HUD_W, 40), (50, 205, 50), 2)
        cv2.putText(hud, "GREENWATCH-1 GIS ORBITAL TERMINAL", (25, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(hud, f"CENARIO: {active_scenario['title'].upper()}", (460, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 205, 50), 1, cv2.LINE_AA)

        # 6. Painel de Comandos (Rodapé)
        cv2.rectangle(hud, (0, HUD_H - 40), (HUD_W, HUD_H), (20, 20, 20), -1)
        cv2.line(hud, (0, HUD_H - 40), (HUD_W, HUD_H - 40), (50, 50, 50), 1)
        cv2.putText(hud, "Teclas: [1, 2, 3] Chavear Satelite | [S] Salvar Visada | [Q] Sair", (25, HUD_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Indicador de IA ativa ou simulada no rodapé
        detector_status = "YOLOv8 ATIVO (Modelo Real)" if YOLO_AVAILABLE else "YOLOv8 SIMULADO (Aguardando pesos)"
        det_color = (0, 255, 0) if YOLO_AVAILABLE else (0, 190, 255)
        cv2.putText(hud, detector_status, (HUD_W - 350, HUD_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, det_color, 1, cv2.LINE_AA)

        # 7. Desenha Linhas de Varredura Estilo Radar/Satélite sobre o feed
        # Adiciona uma grade orbital fina de satélite (cruzamento de coordenadas)
        for i in range(1, 4):
            # Linha vertical
            cv2.line(hud, (40 + i*160, 40), (40 + i*160, 680), (120, 255, 120), 1)
            # Linha horizontal
            cv2.line(hud, (40, 40 + i*160), (680, 40 + i*160), (120, 255, 120), 1)
            
        # Linha laser de scanner em movimento
        self.scan_line_y = (self.scan_line_y + 8) % VIEW_H
        cv2.line(hud, (40, 40 + self.scan_line_y), (680, 40 + self.scan_line_y), (0, 255, 0), 1)

        # 8. Desenha as caixas do YOLO no Feed do HUD
        for det in detections:
            rx, ry, rw, rh = det["bbox"]
            label = det["class"].upper()
            conf = det["confidence"]
            
            # Ajusta para a coordenada local do HUD (deslocamento x=40, y=40)
            hx = rx + 40
            hy = ry + 40
            
            color = (0, 0, 255) if "fogo" in det["class"] else (0, 200, 255)
            
            # Desenha bounding box
            cv2.rectangle(hud, (hx, hy), (hx + rw, hy + rh), color, 2)
            
            # Rótulo
            lbl = f"{label} {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(hud, (hx, hy - 18), (hx + tw + 6, hy), color, -1)
            cv2.putText(hud, lbl, (hx + 3, hy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Desenha retículo central
            cx = hx + rw // 2
            cy = hy + rh // 2
            cv2.drawMarker(hud, (cx, cy), color, cv2.MARKER_TILTED_CROSS, 10, 1)

        # 9. Flashing red border on emergency
        has_fire = any("fogo" in d["class"] for d in detections)
        if has_fire and int(time.time() * 2.5) % 2 == 0:
            cv2.rectangle(hud, (40, 40), (680, 680), (0, 0, 255), 4)
            cv2.rectangle(hud, (info_x - 10, 480), (HUD_W - 30, 680), (0, 0, 255), 2)
            cv2.putText(hud, "EMERGENCIA AMBIENTAL", (info_x, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

        return hud

# ==============================================================================
# LOOP ORBITAL PRINCIPAL
# ==============================================================================
def main():
    print("=" * 70)
    print("      GREENWATCH CV - RASTREAMENTO ORBITAL DE INCENDIOS COM YOLOv8")
    print("=" * 70)
    
    # Inicializa componentes
    detector = SatelliteYoloDetector()
    hud_drawer = GISDashboardHUD()
    
    current_key = "1"
    active_scenario = SCENARIOS[current_key]
    
    # Carrega imagem de satélite inicial
    def load_satellite_image(sc_cfg):
        path = os.path.join(SATELLITE_DIR, sc_cfg["name"])
        if not os.path.exists(path):
            print(f"[ERROR] Imagem {path} nao encontrada. Criando fundo cinza de backup.")
            backup = np.ones((1200, 1600, 3), dtype=np.uint8) * 60
            cv2.putText(backup, "IMAGEM INDISPONIVEL", (300, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (180, 180, 180), 3)
            return backup
        
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Nao foi possivel ler/decodificar a imagem {path}. Criando fundo de backup.")
            backup = np.ones((1200, 1600, 3), dtype=np.uint8) * 60
            cv2.putText(backup, "ERRO DE LEITURA", (300, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (180, 180, 180), 3)
            return backup
            
        # Garante alta resolução 1600x1200 para varredura suave
        return cv2.resize(img, (1600, 1200))

    sat_image = load_satellite_image(active_scenario)
    img_h, img_w = sat_image.shape[:2]

    # Coordenadas iniciais do Varredor Orbital (Sweep)
    sweep_x = 0
    sweep_y = 0
    dir_x = 1  # 1 = Direita, -1 = Esquerda
    dir_y = 1  # 1 = Baixo, -1 = Cima
    
    # Histerese de alerta
    fire_frames = 0
    smoke_frames = 0
    
    print("[SISTEMA] Varredura Orbital iniciada com sucesso. Janela OpenCV aberta.")

    while True:
        # 1. Movimentação sistemática da varredura orbital (zigzag suave)
        step_x = 4 # Velocidade orbital no eixo X (pixels por frame)
        sweep_x += dir_x * step_x
        
        # Limites no eixo X (Viewport 640x640 na imagem 1600x1200)
        max_x = img_w - VIEW_W
        max_y = img_h - VIEW_H
        
        if sweep_x >= max_x:
            sweep_x = max_x
            dir_x = -1
            # Incrementa o eixo Y ao atingir a borda lateral
            sweep_y += 70 * dir_y
        elif sweep_x <= 0:
            sweep_x = 0
            dir_x = 1
            sweep_y += 70 * dir_y
            
        # Limites no eixo Y
        if sweep_y >= max_y:
            sweep_y = max_y
            dir_y = -1
        elif sweep_y <= 0:
            sweep_y = 0
            dir_y = 1

        # 2. Recorta o frame de visualização ativo (640x640)
        viewport_frame = sat_image[sweep_y:sweep_y+VIEW_H, sweep_x:sweep_x+VIEW_W].copy()
        
        # 3. Executa inferência do YOLOv8 (Real ou Simulada)
        detections = detector.detect(viewport_frame, current_key, sweep_x, sweep_y)
        
        # 4. Histerese e Processamento de Alertas GIS
        has_fire = any("fogo" in d["class"] for d in detections)
        has_smoke = any("fumaca" in d["class"] for d in detections)
        
        # Mapeamento dinâmico de coordenadas GPS do foco detectado
        # Cada pixel representa aproximadamente 10 metros no solo
        if has_fire:
            fire_frames += 1
            if fire_frames >= 4: # Confirmado por 4 frames consecutivos
                fire_det = next(d for d in detections if "fogo" in d["class"])
                rx, ry, rw, rh = fire_det["bbox"]
                
                # Conversão para escala global e latitude/longitude
                abs_x = sweep_x + rx + rw // 2
                abs_y = sweep_y + ry + rh // 2
                
                target_lat = active_scenario["lat"] - (abs_y / img_h) * 0.05
                target_lon = active_scenario["lon"] + (abs_x / img_w) * 0.05
                
                # Cálculo da área baseado no tamanho do bounding box (1 pixel = 100 m² = 0.01 hectares)
                area_ha = (rw * rh * 100) / 10000.0
                
                # Grava alerta estruturado
                alert = GISAlertLogger.log_alert("FOGO", fire_det["confidence"], target_lat, target_lon, area_ha, active_scenario["region"])
                hud_drawer.add_alert(alert)
        else:
            fire_frames = 0
            
        if has_smoke:
            smoke_frames += 1
            if smoke_frames >= 6: # Confirmado por 6 frames
                smoke_det = next(d for d in detections if "fumaca" in d["class"])
                rx, ry, rw, rh = smoke_det["bbox"]
                
                abs_x = sweep_x + rx + rw // 2
                abs_y = sweep_y + ry + rh // 2
                
                target_lat = active_scenario["lat"] - (abs_y / img_h) * 0.05
                target_lon = active_scenario["lon"] + (abs_x / img_w) * 0.05
                area_ha = (rw * rh * 100) / 10000.0
                
                alert = GISAlertLogger.log_alert("FUMACA", smoke_det["confidence"], target_lat, target_lon, area_ha, active_scenario["region"])
                hud_drawer.add_alert(alert)
        else:
            smoke_frames = 0

        # 5. Renderiza a tela do Dashboard HUD
        hud_frame = hud_drawer.draw(viewport_frame, detections, active_scenario, sweep_x, sweep_y, img_w, img_h)
        
        # 6. Exibe na tela
        cv2.imshow(WINDOW_NAME, hud_frame)
        
        # 7. Entrada de teclado
        key = cv2.waitKey(15) & 0xFF  # Cerca de 60 FPS suavizado
        
        # Alternância de Cenários de Satélite
        if key in [ord('1'), ord('2'), ord('3')]:
            current_key = chr(key)
            active_scenario = SCENARIOS[current_key]
            sat_image = load_satellite_image(active_scenario)
            # Reseta coordenadas de varredura
            sweep_x, sweep_y = 0, 0
            dir_x, dir_y = 1, 1
            print(f"[SATELITE] Chaveando para o cenario: {active_scenario['title']}")
            
        # Tecla S: Salva Snapshot manual do frame e HUD
        elif key == ord('s') or key == ord('S'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SNAPSHOTS_DIR}/orbital_scan_{timestamp}.jpg"
            cv2.imwrite(filename, hud_frame)
            print(f"[SNAPSHOT] Captura orbital salva em: {filename}")
            
        # Tecla Q ou Esc: Sai da aplicação
        elif key == ord('q') or key == ord('Q') or key == 27:
            print("[SISTEMA] Desconectando do terminal orbital...")
            break

    cv2.destroyAllWindows()
    print("=" * 70)
    print("       GREENWATCH CV - CONEXÃO ORBITAL ENCERRADA COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    main()
