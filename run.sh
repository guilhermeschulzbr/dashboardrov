#!/bin/bash


# Ativa ambiente virtual se existir
if [ -d "venv" ]; then
    source venv/bin/activate
fi


# Instala dependências se necessário
pip install -r requirements.txt


# Executa o dashboard
streamlit run main.py --server.port 8501 --server.address localhost