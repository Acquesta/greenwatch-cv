# GreenWatch CV - Sensoriamento Orbital e Controle Gestual de IA

Este projeto apresenta uma solução inovadora de Visão Computacional para o monitoramento territorial de biomas brasileiros (Amazônia, Pantanal e Cerrado). Ele une a inferência inteligente de redes neurais (YOLOv8) com a interatividade humana de sensoriamento remoto, operado por gestos de mãos capturados via webcam usando a biblioteca Google MediaPipe.

Toda a lógica e a interface gráfica tática estão integradas e unificadas em um único arquivo principal: `main.py`.

---

## 📖 1. Descrição da Solução

O **GreenWatch CV** é um software de processamento digital de imagens que simula o controle interativo de um satélite ou drone tático de busca e salvamento (SAR). O painel gráfico (HUD HUD) apresenta os seguintes recursos principais:

* **Rastreamento por Palma (Landmark 9)**: O sistema detecta a mão do operador através da webcam e projeta a estrutura óssea anatômica em tempo real em uma janela Picture-in-Picture (PIP). O centro da palma (articulação do dedo médio, Landmark 9) orienta a posição contínua e dinâmica da mira e da lente de varredura verde sobre o mapa.
* **Lente de Zoom Mecânico Dinâmico**:
  * 🖐️ **Mão Aberta (5 dedos)**: A lente verde no mapa se contrai de forma fluida, operando um **Zoom In Máximo (3.75x de aproximação)** sobre a área de desastre (exibindo um recorte de $96 \times 72$ pixels ampliado no painel).
  * ✊ **Punho Fechado (0 dedos)**: A lente verde se expande, operando um **Zoom Out Amplo (1.5x de visão geral)** sobre os arredores (recorte de $240 \times 180$ pixels).
  * **Interpolação Mecânica Suave**: Para conferir realismo físico, as dimensões da lente de zoom são recalculadas a cada frame usando interpolação exponencial suave, imitando a movimentação mecânica de uma lente real de satélite.
* **Detector Inteligente Híbrido (YOLOv8 + Mock Georreferenciado)**:
  * A inferência inteligente de detecção roda **exclusivamente dentro do quadro ampliado**, otimizando performance.
  * O sistema tenta carregar os pesos da rede YOLOv8 (`best.pt`). Este arquivo de pesos é baseado no modelo pré-treinado especializado em identificação de chamas e incêndios florestais obtido diretamente do HuggingFace: [SalahALHaismawi/yolov26-fire-detection](https://huggingface.co/SalahALHaismawi/yolov26-fire-detection).
  * Quando carregado, realiza a classificação em tempo real das classes de incêndio ("fogo") e fumaça.
  * Se o arquivo `best.pt` não estiver disponível no diretório raiz, o sistema aciona de forma transparente um **Detector Mock Georreferenciado Proporcional** de altíssima fidelidade que escala as coordenadas dos alvos de fogo/fumaça de acordo com o nível atual de aproximação do zoom e coordenadas geográficas ativas da lente.
* **Troca Gestual de Biomas (Victory Sign)**:
  * ✌️ **Sinal de Vitória (2 dedos)**: Segurar este gesto por **40 frames (~1.3 segundos)** projeta uma barra de progresso no HUD. Ao completar, altera ciclicamente o canal orbital, carregando novas imagens de satélite reais de alta resolução espacial: **Amazônia** (Codajás-AM) ➡️ **Pantanal** (Corumbá-MS) ➡️ **Cerrado** (Querência-MT).
* **Alarme Estroboscópico de Emergência (5s)**:
  * Caso um foco classificado como **"fogo"** permaneça dentro da lente de zoom por **5 segundos contínuos ou mais**, a central de controle entra em modo de emergência ativa: o HUD passa a **piscar uma borda vermelha espessa (6px)** e exibe um **banner central de alerta máximo de incêndio**.

---

## 🛠️ 2. Bibliotecas Utilizadas

O sistema é construído sobre quatro pilares essenciais de processamento de imagens e IA:

1. **MediaPipe (`mediapipe`)**: Framework do Google usado para rastreamento de landmarks da mão no modo de vídeo síncrono e cálculo dinâmico da quantidade de dedos estendidos.
2. **Ultralytics YOLO (`ultralytics`)**: Execução e inferência de redes neurais artificiais YOLOv8 para identificação e bounding boxes rápidas de plumas de fumaça e focos de calor.
3. **OpenCV (`opencv-python`)**: Leitura matricial de imagens de satélite, captura de vídeo da webcam física, redimensionamentos bilineares de crop de zoom e renderização gráfica de todos os painéis e textos do HUD.
4. **NumPy (`numpy`)**: Manipulação matemática rápida de matrizes de imagens, conversões de coordenadas de pixels e cálculos de proporção geométrica da lupa.

---

## 🚀 3. Instruções Básicas de Execução

### Passo 1: Instalação das Dependências
Abra o seu terminal na pasta raiz do projeto e certifique-se de instalar as dependências necessárias através do gerenciador de pacotes do Python:
```bash
pip install -r requirements.txt
```

### Passo 2: Execução da Central
Com a webcam ativa conectada ao computador, execute o arquivo principal da aplicação:
```bash
python main.py
```
> [!NOTE]
> Na primeira inicialização, o script detectará automaticamente a ausência do arquivo `hand_landmarker.task` e fará o download do modelo oficial do Google de forma automática e transparente diretamente no diretório raiz do projeto.

### Passo 3: Interação
* **Gesto 🖐️ (5 Dedos)**: Aplica aproximação máxima de Zoom In.
* **Gesto ✊ (0 Dedos)**: Afasta para obter visão ampla (Zoom Out).
* **Gesto ✌️ (2 Dedos)**: Segure para alternar ciclicamente o bioma.
* **Teclas `Q` ou `ESC`**: Fecha a janela e finaliza a transmissão do satélite.

---

## 👥 4. Integrantes do Grupo

Substitua as informações abaixo com os dados dos respectivos integrantes do grupo para a entrega da sua Global Solution:

* **[Nome do Integrante 1]** - RM: [XXXXX]
* **[Nome do Integrante 2]** - RM: [XXXXX]
* **[Nome do Integrante 3]** - RM: [XXXXX]
* **[Nome do Integrante 4]** - RM: [XXXXX]
* **[Nome do Integrante 5]** - RM: [XXXXX]
