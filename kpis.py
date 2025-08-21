# -*- coding: utf-8 -*-
"""Componente de KPIs (Key Performance Indicators)"""


import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from utils.formatters import BrazilianFormatter
from config.settings import Config


class KPIComponent:
    """Componente para renderização de KPIs"""
    
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config
        self.formatter = BrazilianFormatter()
    
    def calculate_main_kpis(self) -> Dict[str, Any]:
        """Calcula os KPIs principais"""
        kpis = {}
        
        # Total de passageiros
        if "Passageiros" in self.df.columns:
            kpis["total_passageiros"] = self.df["Passageiros"].sum()
        else:
            kpis["total_passageiros"] = 0
        
        # Total de viagens
        kpis["total_viagens"] = len(self.df)
        
        # Distância total
        if "Distancia" in self.df.columns:
            kpis["distancia_total"] = self.df["Distancia"].sum()
        else:
            kpis["distancia_total"] = 0
        
        # Média de passageiros por viagem
        if kpis["total_viagens"] > 0:
            kpis["media_pax_viagem"] = kpis["total_passageiros"] / kpis["total_viagens"]
        else:
            kpis["media_pax_viagem"] = 0
        
        # Veículos únicos
        if "Numero Veiculo" in self.df.columns:
            kpis["veiculos_unicos"] = self.df["Numero Veiculo"].nunique()
        else:
            kpis["veiculos_unicos"] = 0
        
        # Linhas ativas
        if "Nome Linha" in self.df.columns:
            kpis["linhas_ativas"] = self.df["Nome Linha"].nunique()
        else:
            kpis["linhas_ativas"] = 0
        
        # IPK (Índice de Passageiros por Quilômetro)
        if kpis["distancia_total"] > 0:
            kpis["ipk"] = kpis["total_passageiros"] / kpis["distancia_total"]
        else:
            kpis["ipk"] = 0
        
        return kpis
    
    def calculate_financial_kpis(self) -> Dict[str, Any]:
        """Calcula os KPIs financeiros"""
        kpis = {}
        
        # Total de pagantes
        cols_pagantes = [col for col in self.config.COLS_PAGANTES if col in self.df.columns]
        if cols_pagantes:
            kpis["total_pagantes"] = self.df[cols_pagantes].sum().sum()
        else:
            kpis["total_pagantes"] = 0
        
        # Total de gratuidades
        if "Quant Gratuidade" in self.df.columns:
            kpis["total_gratuidades"] = self.df["Quant Gratuidade"].sum()
        else:
            kpis["total_gratuidades"] = 0
        
        # Receita tarifária
        kpis["receita_tarifaria"] = kpis["total_pagantes"] * self.config.TARIFA_USUARIO
        
        # Subsídio total
        kpis["subsidio_total"] = kpis["total_pagantes"] * self.config.SUBSIDIO_PAGANTE
        
        # Receita total
        kpis["receita_total"] = kpis["receita_tarifaria"] + kpis["subsidio_total"]
        
        # Custo público por passageiro
        total_pax = kpis.get("total_passageiros", 0) or self.df["Passageiros"].sum() if "Passageiros" in self.df.columns else 0
        if total_pax > 0:
            kpis["custo_publico_pax"] = kpis["subsidio_total"] / total_pax
        else:
            kpis["custo_publico_pax"] = 0
        
        return kpis
    
    def render_main_kpis(self):
        """Renderiza os KPIs principais"""
        kpis = self.calculate_main_kpis()
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "👥 Passageiros",
                self.formatter.format_integer(kpis["total_passageiros"])
            )
        
        with col2:
            st.metric(
                "🧭 Viagens",
                self.formatter.format_integer(kpis["total_viagens"])
            )
        
        with col3:
            st.metric(
                "🛣️ Distância (km)",
                self.formatter.format_number(kpis["distancia_total"], 1)
            )
        
        with col4:
            st.metric(
                "📈 Média pax/viagem",
                self.formatter.format_number(kpis["media_pax_viagem"], 2)
            )
        
        with col5:
            st.metric(
                "🚌 Veículos",
                self.formatter.format_integer(kpis["veiculos_unicos"])
            )
        
        with col6:
            st.metric(
                "🧵 Linhas ativas",
                self.formatter.format_integer(kpis["linhas_ativas"])
            )
    
    def render_financial_kpis(self):
        """Renderiza os KPIs financeiros"""
        kpis = self.calculate_financial_kpis()
        
        st.subheader("💰 Indicadores Financeiros")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                "Pagantes",
                self.formatter.format_integer(kpis["total_pagantes"])
            )
        
        with col2:
            st.metric(
                "Gratuidades",
                self.formatter.format_integer(kpis["total_gratuidades"])
            )
        
        with col3:
            st.metric(
                "Receita Tarifária",
                self.formatter.format_currency(kpis["receita_tarifaria"])
            )
        
        with col4:
            st.metric(
                "Subsídio Total",
                self.formatter.format_currency(kpis["subsidio_total"])
            )
        
        with col5:
            st.metric(
                "Custo Público/Pax",
                self.formatter.format_currency(kpis["custo_publico_pax"])
            )
        
        with col6:
            st.metric(
                "Receita Total",
                self.formatter.format_currency(kpis["receita_total"])
            )
    
    def render_custom_kpi(self, 
                         label: str,
                         value: float,
                         format_type: str = "number",
                         delta: Optional[float] = None):
        """
        Renderiza um KPI customizado
        
        Args:
            label: Rótulo do KPI
            value: Valor do KPI
            format_type: Tipo de formatação (number, currency, percentage, integer)
            delta: Valor de variação (opcional)
        """
        if format_type == "currency":
            formatted_value = self.formatter.format_currency(value)
        elif format_type == "percentage":
            formatted_value = self.formatter.format_percentage(value)
        elif format_type == "integer":
            formatted_value = self.formatter.format_integer(value)
        else:
            formatted_value = self.formatter.format_number(value)
        
        if delta is not None:
            st.metric(label, formatted_value, delta=f"{delta:+.1f}%")
        else:
            st.metric(label, formatted_value)
