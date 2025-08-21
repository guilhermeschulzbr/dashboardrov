# -*- coding: utf-8 -*-
"""Configurações e constantes do sistema"""


import streamlit as st
from typing import List, Dict, Any


class Config:
    """Configurações principais do sistema"""
    
    # Parâmetros de jornada
    JORNADA_HORAS = 7
    JORNADA_MINUTOS = 20
    JORNADA_TOTAL_MIN = JORNADA_HORAS * 60 + JORNADA_MINUTOS
    
    # Parâmetros financeiros (valores padrão)
    TARIFA_USUARIO = 2.00
    SUBSIDIO_PAGANTE = 4.22
    
    # Limiares de alerta
    DISTANCIA_ALTA_KM = 20.0
    PASSAGEIROS_BAIXOS = 5
    
    # Colunas esperadas
    COLS_MOTORISTA = [
        "Cobrador/Operador",
        "Matricula",
        "Matrícula",
        "Nome Motorista",
        "CPF Motorista",
        "ID Motorista"
    ]
    
    COLS_VEICULO = [
        "Numero Veiculo",
        "Número Veículo",
        "Codigo Veiculo",
        "Código Veículo",
        "Placa"
    ]
    
    COLS_LINHA = [
        "Nome Linha",
        "Linha",
        "Codigo Linha",
        "Código Linha"
    ]
    
    COLS_DATA = [
        "Data Coleta",
        "Data",
        "DataColeta"
    ]
    
    COLS_HORARIO_INICIO = [
        "Data Hora Inicio Operacao",
        "Data Hora Início Operação",
        "Inicio Operacao",
        "Início Operação"
    ]
    
    COLS_HORARIO_FIM = [
        "Data Hora Final Operacao",
        "Data Hora Final Operação",
        "Fim Operacao",
        "Final Operação"
    ]
    
    COLS_PASSAGEIROS = [
        "Passageiros",
        "Qtd Passageiros",
        "Quantidade Passageiros",
        "Total Passageiros"
    ]
    
    COLS_DISTANCIA = [
        "Distancia",
        "Distância",
        "KM",
        "Quilometragem"
    ]
    
    COLS_PAGANTES = [
        "Quant Inteiras",
        "Quant Passagem",
        "Quant Passe",
        "Quant Vale Transporte"
    ]
    
    COLS_GRATUIDADE = [
        "Quant Gratuidade",
        "Gratuidades",
        "Gratuidade"
    ]
    
    COLS_INTEGRACAO = [
        "Quant Passagem Integracao",
        "Quant Passe Integracao",
        "Quant Vale Transporte Integracao"
    ]
    
    # Configurações de cache
    CACHE_TTL = 3600  # 1 hora
    
    # Configurações de visualização
    MAX_ROWS_DISPLAY = 100
    CHART_HEIGHT = 400
    
    # Cores para gráficos
    COLOR_SCHEME = {
        "primary": "#1f77b4",
        "success": "#2ca02c",
        "warning": "#ff7f0e",
        "danger": "#d62728",
        "info": "#17a2b8"
    }


class PageConfig:
    """Configurações da página Streamlit"""
    
    @staticmethod
    def setup():
        """Configura a página do Streamlit"""
        st.set_page_config(
            page_title="Dashboard ROV - Operação",
            page_icon="🚌",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # CSS customizado
        st.markdown("""
            <style>
            .main {
                padding-top: 2rem;
            }
            .stMetric {
                background-color: #f0f2f6;
                padding: 10px;
                border-radius: 5px;
                margin: 5px 0;
            }
            </style>
        """, unsafe_allow_html=True)
