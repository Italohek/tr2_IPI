# Processamento Digital de Imagens

Este repositório tem a implementação de dois projetos desenvolvidos para a disciplina IPI, utilizando Python e bibliotecas da área de visão computacional (foco em scikit-learn).

## Projeto 1 – Detecção de Tumor Cerebral

Este projeto realiza a detecção automática de uma possível região de tumor em uma imagem de ressonância magnética utilizando técnicas clássicas de processamento de imagens.

### Etapas do processamento

- Conversão da imagem para escala de cinza;
- Aplicação de filtro Gaussiano;
- Aplicação de filtro da Mediana;
- Limiarização utilizando Multi-Otsu;
- Operações morfológicas (Opening e Closing);
- Remoção de componentes conectados às bordas;
- Rotulagem de componentes conexos;
- Seleção da região de interesse com base na área e na propriedade *solidity*;
- Extração e exibição do contorno do tumor.

---

## Projeto 2 – Segmentação por K-Means

Este projeto realiza a segmentação da imagem **onion.jpg** utilizando o algoritmo **K-Means (Max-Lloyd)** da biblioteca Scikit-Learn.

### Etapas do processamento

- Leitura da imagem;
- Conversão da imagem em uma matriz de pixels RGB;
- Teste de diferentes valores de **K** (2 a 10);
- Aplicação do algoritmo K-Means;
- Avaliação dos agrupamentos utilizando o **Silhouette Score**;
- Seleção automática do melhor número de clusters;
- Reconstrução da imagem utilizando as cores médias de cada cluster;
- Exibição das segmentações, do gráfico do Silhouette Score e do resultado final.

---

## Tecnologias utilizadas

- Python 3
- NumPy
- Matplotlib
- Scikit-Image
- Scikit-Learn

---

## Estrutura do projeto

```
assets/
├── brain.jpg
└── onion.jpg

output/
└── (imagens geradas)

src/
├── tumor_detection.py
└── kmeans_segmentation.py

Relatório.pdf
README.md
```

---

## Como executar

Garanta que você tenha git instalado e prossiga com a clonagem do repositório.

```bash
git clone https://github.com/Italohek/tr2_IPI
```

Certifique-se que tenha o Python 3.14 instalado e crie um ambiente virtual.

```bash
pip -m venv .venv
```
Ative o ambiente virtual.

```bash
.\venv\Scripts\activate
```

Agora instale as dependências dentro do ambiente.
```bash
pip install -r requirements.txt
```

Execute o projeto desejado:

```bash
cd .\src\
py .\main.py ou py .\onion.py
```

## Autor

Italo Braga de Paulo