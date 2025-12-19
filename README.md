# Face Liveness Detection Simples (Anti-Spoofing)

Projeto de **Biometria / Análise Forense Digital** focado em detecção simples de *liveness* facial, com o objetivo de impedir que alguém desbloqueie um sistema usando **foto** ou **vídeo** em vez do rosto real.

Neste estágio (~40% do projeto), está implementado o **módulo de detecção de piscar (blink)** baseado no **Eye Aspect Ratio (EAR)**.

## Objetivo

Desenvolver um protótipo de produto/aplicação capaz de:

- Detectar a presença de um rosto em frente à câmera.
- Observar o comportamento dos olhos ao longo do tempo.
- Verificar se houve pelo menos um **piscar natural** dentro de uma janela de tempo.
- Utilizar o piscar como evidência de que a face é de um **indivíduo real** (não apenas uma foto estática).

Em versões futuras, o projeto será estendido com:

- Análise de **variação de brilho/reflexo** no rosto, para diferenciar **rostos 3D** de ataques 2D (foto, tela de celular).
- Conjunto de experimentos comparando:
  - Método A: apenas EAR (blink).
  - Método B: EAR + brilho/reflexo.

## Tecnologias utilizadas

- Python
- OpenCV (captura de vídeo, visualização)
- MediaPipe Face Mesh (detecção de face e landmarks)
- NumPy (cálculos geométricos)

## Estrutura do projeto

```text
face-liveness-detection/
├── README.md
├── requirements.txt
└── src/
    ├── main.py
    ├── config.py
    ├── detection/
    │   ├── __init__.py
    │   └── face_and_landmarks.py
    ├── features/
    │   ├── __init__.py
    │   └── ear_blink.py
    └── utils/
        ├── __init__.py
        └── video_io.py
```
## Face and Landmarks (src/detection/face_and_landmarks.py)

A partir desse código é possível detectar a piscadas, analisar a luminosidade e liveness, pois ele retorna os 468 pontos faciais
que usamos para calcular o EAR, desenhar os olhos na tela e também recortar a região do rosto para o módulo de brilho.

- Responsável por inicializar o modelo de detecção facial:
  Cria um objeto do MediaPipe Face Mesh 
  Define as configurações (número máximo de faces, nível de confiança e nível de rastreamento)

- Recebe cada frame da webcam:
  O detector recebe uma imagem OpenCV -> Converte para RGB -> Envia pro modelo

- Fluxo:
    -> Detecta se existe um rosto na imagem
    -> Extrai os landmarks do rosto (468 pontos anatômicos em coordenadas x,y,z)
    -> Converte landmarks em coordernadas pixel (Opcional em outros casos, mas o nosso módulo de brilho precisa recortar o rosto usando coordenadas reais)
    -> Retorna os landmarks para o pipeline

## Brightness (src/features/brightness.py)

- Implementa o módulo de análise fotométrica (brilho/reflexo) usado para detectar se a superfície mostrada à câmera se 
comporta como um rosto real (3D) ou uma superfície artificial (2D)
- Extrai a Região do Rosto (ROI)
  - O módulo recebe um frame BGR (OpenCV) e os landmarks faciais detectados pelo MediaPipe
  - A partir dos landmarks, é encontrado as coordenadas mínimas e máximas do rosto no frame
e recorta a região correspondente ao rosto e converte a ROI para escala de cinza. 
- Cálculo do Brilho por Frame
  - Média do brilho (mean)

## Ear Blink (src/features/brightness.py)
