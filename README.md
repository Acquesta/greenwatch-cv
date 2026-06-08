# GreenWatch CV - Sensoriamento Orbital e Controle Gestual de IA

O **GreenWatch CV** é uma solução de Visão Computacional para o monitoramento de biomas brasileiros (Amazônia, Pantanal e Cerrado). O sistema simula o controle interativo de um satélite, unindo a detecção de redes neurais (YOLOv8) com sensoriamento remoto operado por gestos de mão (capturados via webcam usando Google MediaPipe).

Toda a aplicação está centralizada no arquivo `main.py`.

---

## 📖 1. Funcionalidades Principais

* **Rastreamento Gestual**: O centro da palma da mão guia a mira e a lente de varredura no mapa em tempo real.
* **Zoom Dinâmico (Lente Mecânica)**:
  * 🖐️ **Mão Aberta (5 dedos)**: Zoom In sobre a área.
  * ✊ **Punho Fechado (0 dedos)**: Zoom Out para visão geral.
* **Detecção Inteligente (YOLOv8)**: Identifica focos de incêndio e fumaça em tempo real utilizando o modelo `best.pt`. Caso o modelo não seja encontrado, um sistema simulado de detecção (mock) de alta fidelidade é acionado automaticamente.
* **Troca de Biomas**:
  * ✌️ **Sinal de Vitória (2 dedos)**: Segure o gesto por alguns frames para alternar as imagens de satélite entre Amazônia, Pantanal e Cerrado.
* **Alerta de Emergência**: Focos de incêndio mantidos no centro da lente por mais de 5 segundos contínuos ativam um modo de emergência com alertas visuais no painel (HUD).

---

## 🛠️ 2. Tecnologias Utilizadas

* **MediaPipe**: Rastreamento avançado de mãos e gestos.
* **Ultralytics (YOLO)**: Rede neural para detecção de fogo/fumaça.
* **OpenCV**: Renderização, interface HUD, leitura de webcam e redimensionamento de imagens.
* **NumPy**: Cálculos matemáticos e manipulação de matrizes de imagens.

---

## 🚀 3. Como Executar

### Pré-requisitos
Instale as dependências executando no terminal:
```bash
pip install -r requirements.txt
```

### Execução
Com uma webcam ativa conectada, inicie a aplicação:
```bash
python main.py
```

### Comandos de Interação
* **🖐️ 5 Dedos**: Aproximar (Zoom In Máximo).
* **✊ 0 Dedos**: Afastar (Zoom Out Amplo).
* **✌️ 2 Dedos (Sinal da Vitória)**: Segure para alternar o bioma.
* **Teclas `Q` ou `ESC`**: Sair do sistema.

---

## 👥 4. Integrantes do Grupo

* **Alexia Ramalho** - RM: 558385
* **Enzo Real** - RM: 557943
* **Gustavo Pasquini** - RM: 555454
* **Hellen Silva** - RM: 559008
* **Lorenzo Acquesta** - RM: 557397
