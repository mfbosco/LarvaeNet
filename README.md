# LarvaeNet

Projeto de rede neural convolucional para detecção e classificação de larvas de helmintos em imagens microscópicas.

## Descrição do Projeto

Este projeto visa desenvolver e avaliar modelos de redes neurais convolucionais (ConvNets) para a classificação de imagens microscópicas em duas classes:
- **000001**: Larvas de helmintos
- **000002**: Impurezas (restos alimentares)

## Objetivos

### 1. Modelo Básico Inicial

Desenvolvimento de um modelo ConvNet treinado do zero com as seguintes características:

- **Divisão dos dados**: 50% treino, 20% validação, 30% teste
- **Métricas de avaliação**:
  - Acurácia
  - Perda (loss)
  - Kappa de Cohen
- **Visualizações**:
  - Ativações de camadas convolucionais salvas como imagens PNG normalizadas (0-255)
  - Projeções tSNE/UMAP da última camada convolucional e das camadas densas
  - Regiões de atenção em imagens com predição correta
- **Ajuste de hiperparâmetros** baseado nas visualizações e métricas de validação

### 2. Arquiteturas Avançadas

Exploração de diferentes blocos arquiteturais para melhorar o desempenho do modelo básico:

- **Blocos Residuais** (ResNet-style)
- **Blocos de Inception**
- **Blocos de Atenção**: CBAM (Convolutional Block Attention Module) ou SE (Squeeze-and-Excitation)

Comparação sistemática com o modelo básico para verificar ganhos de desempenho.

### 3. Transfer Learning

Utilização de codificadores pré-treinados na ImageNet como baseline de referência:

- **VGG-16**: Arquitetura clássica com camadas convolucionais sequenciais
- **ResNet-50**: Arquitetura com blocos residuais

Comparação do desempenho dos modelos treinados do zero versus modelos pré-treinados.


## Requisitos

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- umap-learn
- pandas

## Resultados 

- Modelo base funcional com boas métricas de classificação
- Comparação quantitativa entre diferentes arquiteturas
- Análise de transferência de aprendizado vs. treinamento do zero
- Visualizações interpretáveis das decisões do modelo
- Arquivo [`relatorio_final.pdf`](relatorio_final.pdf) contém o relatório final sobre o estudo.
