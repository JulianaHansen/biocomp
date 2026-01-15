# BioCel Computacional

Este projeto implementa utiliza Machine Learning para identificação de células em imagens histológicas de granulomas no formato `.tif`.

## Pré-requisitos

Antes de começar, certifique-se de ter as seguintes ferramentas instaladas no seu sistema:

1.  **Git**: Para versionamento de código. [Download Git](https://git-scm.com/downloads)
2.  **Miniconda** (Recomendado) ou **Anaconda**: Para gerenciamento de ambientes virtuais e bibliotecas científicas. [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)

---

## Instalação e Configuração

Siga estes passos para rodar o projeto em uma nova máquina.

### 1. Clonar o Repositório

Abra seu terminal (ou Anaconda Prompt no Windows) e clone o projeto:

```bash
git clone [link]
cd seu-projeto

### 2. Criar o Ambiente Virtual

Este projeto utiliza um arquivo environment.yml para garantir que todas as dependências (NumPy, OpenCV, Tifffile, etc.) sejam instaladas nas versões corretas.
Rode o comando abaixo para criar o ambiente automaticamente:

```bash
conda env create -f environment.yml

### 3. Ativar o Ambiente

Sempre que for trabalhar neste projeto, você deve ativar o ambiente virtual criado (o nome padrão configurado é bioimagem):

```bash
conda activate bioimagem

## Como Executar

Com o ambiente ativado, você pode rodar os notebooks de análise ou os scripts de processamento.

### Para rodar Jupyter Notebooks:

```bash
jupyter lab

Isso abrirá uma interface no seu navegador. Navegue até a pasta notebooks/ e abra os arquivos .ipynb.

### Para rodar scripts Python:

```bash
python src/main.py

### Fluxo de Desenvolvimento (Manutenção)

Instruções para manter o ambiente sincronizado entre diferentes computadores ou desenvolvedores.

## Você instalou uma nova biblioteca
Se você precisou instalar um pacote novo durante o desenvolvimento:

Instale o pacote:

```bash
conda install [nome do pacote]

Atualize o arquivo de registro (environment.yml): É crucial rodar este comando para salvar a alteração. A flag --no-builds garante que o arquivo funcione em Windows, Linux e Mac.

```bash
conda env export --no-builds > environment.yml

Suba a alteração para o GitHub:

```bash
git add environment.yml
git commit -m "Adiciona biblioteca scipy"
git push