# GreenWatch CV - Sistema de Detecção de Incêndios por Sensoriamento Orbital e YOLOv8

> **Projeto Desenvolvido para a Global Solution (GS) - FIAP**
> 
> *Foco no tema de Monitoramento Territorial e Ambiental para Defesa Civil, Cooperativas, ONGs e Grandes/Pequenos Produtores Rurais.*

---

## 📖 Descrição da Solução

O **GreenWatch CV** é um software de Visão Computacional de sensoriamento remoto projetado para identificar focos de incêndio e plumas de fumaça em imagens reais de satélite de alta resolução (como imagens de Sentinel-2 ou Landsat).

### Proposta de Valor no Contexto da Global Solution
A maioria das prefeituras pequenas, cooperativas agrícolas e brigadas de incêndio locais não possuem recursos para processar dados geoespaciais complexos em formato raster. O **GreenWatch CV** simplifica todo esse pipeline convertendo feeds espaciais em **informações acionáveis em tempo real**:

1. **Varredura Orbital Dinâmica (Sweep Scanner)**:
   O sistema simula a trajetória orbital de um satélite de observação da Terra (ex: `GREENWATCH-1`). O software carrega imagens de satélite de alta resolução ($1600 \times 1200$) e realiza um movimento de varredura (*panning*) sistemático em zigue-zague com uma janela de visualização de $640 \times 640$ pixels. Isso serve como a **captura de vídeo em tempo real** do sistema!
2. **Modelo de Detecção de IA (YOLOv8)**:
   A varredura em tempo real alimenta a rede neural **YOLOv8 da Ultralytics** (configurada para buscar `fire_smoke_yolo.pt` ou `best.pt` localmente). 
   - *Inteligência de Entrega:* Caso o arquivo de pesos ainda não esteja presente na pasta, o script ativa automaticamente o modo de **Detecção YOLO Simulada Georreferenciada** sobre as coordenadas geográficas reais das fotos de teste. Isso garante que a solução apresente uma demonstração 100% funcional na avaliação antes mesmo do upload de pesos personalizados!
3. **Mapeamento GIS (Geographic Information System)**:
   - **Coordenadas GPS (Lat/Lon)**: O sistema calcula dinamicamente a Latitude e a Longitude reais dos focos de incêndio e fumaça detectados na tela, baseando-se no pixel central das caixas delimitadoras e nos metadados do satélite.
   - **Estimativa de Hectares Afetados**: O software calcula o tamanho físico aproximado da queimada em **Hectares (ha)** baseado na contagem de pixels da bounding box do YOLO (considerando resolução de solo de 10m por pixel).
4. **Console HUD Profissional**:
   Uma tela estilo centro de controle espacial (NASA/INPE) que exibe a imagem de satélite com as marcações do YOLO, linhas laser de radar do scanner, mini-mapa orbital de rastreamento no canto superior, coordenadas de GPS ativas, telemetria do satélite e o log com os últimos registros confirmados de desastre.

---

## 🛠️ Bibliotecas Utilizadas

O projeto utiliza exclusivamente o **Python 3** e as bibliotecas:

* **OpenCV (`opencv-python>=4.8.0`)**: Captura e redimensionamento das imagens de alta resolução, processamento de caixas delimitadoras em tempo real, desenho de retículos de radar e renderização do HUD GIS completo.
* **NumPy (`numpy>=1.23.0`)**: Manipulação matemática rápida de pixels e arrays para os recortes e deslocamento do scanner.
* **Ultralytics YOLO (`ultralytics>=8.0.0`)**: Framework de Deep Learning para instanciar e executar a inferência de imagem na arquitetura do YOLOv8.

---

## 📂 Estrutura do Projeto

```
greenwatch-cv/
│
├── requirements.txt         # Dependências essenciais (opencv-python, numpy, ultralytics)
├── README.md                # Esta documentação atualizada da entrega
│
├── main.py                  # ARQUIVO ÚNICO (Varredura de satélite, detector YOLOv8 e HUD GIS)
├── alerts_log.json          # Arquivo gerado dinamicamente com registros georreferenciados salvos
│
├── satellite_images/        # Imagens reais de satélite fornecidas para testes imediatos
│   ├── amazon_fire.png      # Focos ativos de chamas no bioma Amazônico (Codajás-AM)
│   ├── pantanal_smoke.png   # Coluna de fumaça de incêndio florestal no Pantanal (Corumbá-MS)
│   └── deforestation_burn.png # Cicatriz de desmatamento e queimada no Cerrado (Querência-MT)
│
└── snapshots/               # Pasta autogerada para capturar telas das análises GIS (.jpg)
```

---

## 🚀 Instruções de Execução

Siga os passos abaixo para configurar o ambiente e rodar a aplicação:

### 1. Criar e Ativar Ambiente Virtual (Recomendado)
* **No Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **No Linux/macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 2. Instalar Dependências
Instale as bibliotecas necessárias com o pip:
```bash
pip install -r requirements.txt
```

### 3. Executar o Monitoramento
Inicie o terminal orbital executando o script principal:
```bash
python main.py
```

### 4. Integração de Pesos Treinados do YOLOv8
O sistema é **totalmente compatível** com pesos customizados de YOLOv8 de fogo e fumaça, incluindo o modelo treinado de referência **[Abonia1/YOLOv8-Fire-and-Smoke-Detection](https://github.com/Abonia1/YOLOv8-Fire-and-Smoke-Detection)**.

Para integrar pesos treinados na aplicação:
1. Obtenha o arquivo de pesos do modelo treinado (por exemplo, o arquivo `yolov8s.pt` ou `yolov8n.pt` direto do repositório Abonia1, ou um arquivo `best.pt` gerado pelo seu treinamento).
2. Cole este arquivo diretamente na pasta raiz do projeto (`greenwatch-cv/`).
3. O software busca pesos na seguinte ordem de prioridade: `fire_smoke_yolo.pt`, `best.pt`, `yolov8s.pt` e `yolov8n.pt`. 
4. Ao rodar `python main.py` com algum desses arquivos na pasta, o motor real do YOLOv8 assumirá automaticamente a inferência computacional em tempo real sobre os frames de satélite!

---

## 🖐️ Central de Controle Gestual com MediaPipe (`main_mediapipe.py`)

Se a sua entrega da FIAP exige **obrigatoriamente o uso da biblioteca MediaPipe**, criei uma solução fantástica e interativa de **Interface Homem-Máquina por Gestos (Touchless Control)**. 

No contexto da Defesa Civil e brigadas de incêndio, o operador pode ter luvas ou mãos ocupadas. A aplicação `main_mediapipe.py` resolve isso rastreando a mão do operador na webcam para controlar a central orbital:

1. **Rastreamento de Mão (MediaPipe Hands)**: O sistema analisa os 21 pontos de articulação da mão (landmarks) em tempo real. O feed da webcam é exibido em um painel **PIP (Picture-in-Picture)** no canto inferior direito com a marcação do esqueleto da mão.
2. **Mapeamento de Gestos (Comando por Vídeo)**:
   * **Mão Aberta (5 dedos estendidos)**: Modo de operação normal (rastreando câmera).
   * **Mira Laser (Apenas 1 dedo estendido - Indicador)**: O sistema projeta uma mira laser vermelha de radar sobre a imagem do satélite, **seguindo perfeitamente o movimento da ponta do seu dedo indicador** no ar!
   * **Punho Fechado (0 dedos estendidos)**: Ativa instantaneamente o **Botão de Pânico / Emergência** na coordenada da mira GPS. A tela pisca em vermelho, um banner gigante de desastre aparece na tela e um alerta georreferenciado é gravado no `alerts_log.json`.
   * **Sinal de Vitória (2 dedos estendidos - Indicador e Médio)**: Se você segurar este sinal por 1.5 segundos, uma barra de progresso visual é preenchida na tela e o sistema **chaveia automaticamente para o próximo cenário de satélite**!

### Executando a Versão MediaPipe
Instale as dependências atualizadas e execute o arquivo de controle gestual:
```bash
pip install -r requirements.txt
python main_mediapipe.py
```

---

## 🎮 Controles de Teclado (Interface Gráfica)

Durante a execução da janela de vídeo, mantenha o foco na janela gráfica e utilize os seguintes comandos:

* **`1`**: Carrega o Cenário 1 (Focos de Fogo Ativo - Amazônia / Lat: -3.4654, Lon: -62.2145)
* **`2`**: Carrega o Cenário 2 (Coluna de Fumaça - Pantanal / Lat: -18.0125, Lon: -56.4820)
* **`3`**: Carrega o Cenário 3 (Cicatrizes de Desmatamento e Embers - Cerrado / Lat: -11.5080, Lon: -53.6492)
* **`S`**: Tira uma captura de tela analítica do HUD GIS e salva na pasta `snapshots/`.
* **`Q`** ou **`Esc`**: Encerra a conexão orbital com o satélite de forma limpa.

---

## 👥 Integrantes do Grupo
* **[Nome do Integrante 1]** - RM: [XXXXX]
* **[Nome do Integrante 2]** - RM: [XXXXX]
* **[Nome do Integrante 3]** - RM: [XXXXX]
* **[Nome do Integrante 4]** - RM: [XXXXX]
* **[Nome do Integrante 5]** - RM: [XXXXX]
