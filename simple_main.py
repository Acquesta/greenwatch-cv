import cv2
import numpy as np
import os
import json
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

SATELLITE_DIR = "satellite_images"
SNAPSHOTS_DIR = "snapshots"
ALERTS_LOG_FILE = "alerts_log.json"
VIEW_W, VIEW_H = 640, 640
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

SCENARIOS = {
    "1": {"name": "amazon_fire.png", "title": "Codajas (AM) - Amazonia", "lat": -3.4654, "lon": -62.2145},
    "2": {"name": "pantanal_smoke.png", "title": "Corumba (MS) - Pantanal", "lat": -18.0125, "lon": -56.4820},
    "3": {"name": "deforestation_burn.png", "title": "Querencia (MT) - Cerrado", "lat": -11.5080, "lon": -53.6492}
}

SIM_TARGETS = {
    "1": [[1000, 380, 240, 200, "fogo", 0.94], [1020, 420, 320, 280, "fumaca", 0.88]],
    "2": [[800, 600, 520, 450, "fumaca", 0.91]],
    "3": [[480, 720, 180, 150, "fogo", 0.89], [1120, 460, 280, 220, "fumaca", 0.82]]
}

def registrar_alerta(classe, conf, lat, lon, regiao):
    alerta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "threat": classe.upper(),
        "confidence": round(conf * 100, 1),
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "region": regiao
    }
    logs = []
    if os.path.exists(ALERTS_LOG_FILE):
        try:
            with open(ALERTS_LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    if logs:
        ultimos = [l for l in logs if l.get("threat") == alerta["threat"] and l.get("region") == regiao]
        if ultimos:
            ultimo_tempo = datetime.strptime(ultimos[-1]["timestamp"], "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - ultimo_tempo).total_seconds() < 4.0:
                return
    logs.append(alerta)
    with open(ALERTS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
    print(f"[ALERTA] {alerta['timestamp']} - {alerta['threat']} em {regiao} ({alerta['confidence']}%)")

def main():
    model = None
    yolo_loaded = False
    if YOLO_AVAILABLE:
        for peso in ["fire_smoke_yolo.pt", "best.pt", "yolov8s.pt", "yolov8n.pt"]:
            try:
                model = YOLO(peso)
                yolo_loaded = True
                print(f"YOLOv8 carregado: '{peso}'!")
                break
            except Exception:
                continue

    cenario_atual = "1"
    cfg = SCENARIOS[cenario_atual]
    
    def carregar_imagem(sc_cfg):
        path = os.path.join(SATELLITE_DIR, sc_cfg["name"])
        img = cv2.imread(path)
        if img is None:
            backup = np.ones((1200, 1600, 3), dtype=np.uint8) * 60
            cv2.putText(backup, "IMAGEM INDISPONIVEL", (300, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (180, 180, 180), 3)
            return backup
        return cv2.resize(img, (1600, 1200))

    sat_img = carregar_imagem(cfg)
    img_h, img_w = sat_img.shape[:2]

    sweep_x, sweep_y = 0, 0
    dir_x, dir_y = 1, 1
    velocidade = 6
    confirmados = {"fogo": 0, "fumaca": 0}

    while True:
        sweep_x += dir_x * velocidade
        max_x, max_y = img_w - VIEW_W, img_h - VIEW_H
        if sweep_x >= max_x:
            sweep_x, dir_x = max_x, -1
            sweep_y += 75 * dir_y
        elif sweep_x <= 0:
            sweep_x, dir_x = 0, 1
            sweep_y += 75 * dir_y
        if sweep_y >= max_y:
            sweep_y, dir_y = max_y, -1
        elif sweep_y <= 0:
            sweep_y, dir_y = 0, 1

        frame = sat_img[sweep_y:sweep_y+VIEW_H, sweep_x:sweep_x+VIEW_W].copy()
        detections = []

        if yolo_loaded:
            results = model.predict(source=frame, conf=0.25, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                classe = model.names[int(box.cls[0])].lower()
                norm_cls = "fogo" if "fire" in classe or "fogo" in classe else "fumaca"
                detections.append({
                    "class": norm_cls,
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "confidence": conf,
                    "center": (x1 + (x2-x1)//2, y1 + (y2-y1)//2)
                })
        else:
            for target in SIM_TARGETS[cenario_atual]:
                tx_abs, ty_abs, tw, th, cls, conf = target
                ix1, iy1 = max(sweep_x, tx_abs - tw//2), max(sweep_y, ty_abs - th//2)
                ix2, iy2 = min(sweep_x + VIEW_W, tx_abs + tw//2), min(sweep_y + VIEW_H, ty_abs + th//2)
                if (ix2 - ix1) > 40 and (iy2 - iy1) > 40:
                    rx, ry = int(ix1 - sweep_x), int(iy1 - sweep_y)
                    detections.append({
                        "class": cls,
                        "bbox": (rx, ry, int(ix2 - ix1), int(iy2 - iy1)),
                        "confidence": conf,
                        "center": (rx + int(ix2 - ix1)//2, ry + int(iy2 - iy1)//2)
                    })

        tem_fogo = any(d["class"] == "fogo" for d in detections)
        tem_fumaca = any(d["class"] == "fumaca" for d in detections)
        
        for cls, tem in [("fogo", tem_fogo), ("fumaca", tem_fumaca)]:
            if tem:
                confirmados[cls] += 1
                if confirmados[cls] >= 5:
                    det = next(d for d in detections if d["class"] == cls)
                    rx, ry, rw, rh = det["bbox"]
                    abs_cx = sweep_x + rx + rw // 2
                    abs_cy = sweep_y + ry + rh // 2
                    lat_gps = cfg["lat"] - (abs_cy / img_h) * 0.05
                    lon_gps = cfg["lon"] + (abs_cx / img_w) * 0.05
                    registrar_alerta(cls, det["confidence"], lat_gps, lon_gps, cfg["title"])
            else:
                confirmados[cls] = 0

        for det in detections:
            x, y, w, h = det["bbox"]
            cor = (0, 0, 255) if det["class"] == "fogo" else (0, 200, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
            cv2.drawMarker(frame, (x + w//2, y + h//2), cor, cv2.MARKER_CROSS, 10, 1)
            cv2.putText(frame, f"{det['class'].upper()} {det['confidence']*100:.0f}%", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, cor, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (VIEW_W, 45), (15, 15, 15), -1)
        cv2.line(frame, (0, 45), (VIEW_W, 45), (50, 205, 50), 1)
        cv2.putText(frame, f"SATELITE: GREENWATCH-1  |  REGIAO: {cfg['title'].upper()}", (15, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        
        lat_frame = cfg["lat"] - (sweep_y / img_h) * 0.05
        lon_frame = cfg["lon"] + (sweep_x / img_w) * 0.05
        cv2.putText(frame, f"GPS ATIVO: {lat_frame:.5f} S, {lon_frame:.5f} W", (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

        if tem_fogo and int(time.time() * 2) % 2 == 0:
            cv2.rectangle(frame, (VIEW_W - 160, 8), (VIEW_W - 15, 36), (0, 0, 255), -1)
            cv2.putText(frame, "INCENDIO ATIVO", (VIEW_W - 150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("GreenWatch CV - Varredor de Satelite Simplificado", frame)
        key = cv2.waitKey(15) & 0xFF
        
        if key in [ord('1'), ord('2'), ord('3')]:
            cenario_atual = chr(key)
            cfg = SCENARIOS[cenario_atual]
            sat_img = carregar_imagem(cfg)
            sweep_x, sweep_y = 0, 0
            dir_x, dir_y = 1, 1
            print(f"[SATELITE] Cenario: {cfg['title']}")
            
        elif key == ord('s') or key == ord('S'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{SNAPSHOTS_DIR}/simple_scan_{ts}.jpg", frame)
            print("Snapshot Salvo!")
            
        elif key in [ord('q'), ord('Q'), 27]:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
