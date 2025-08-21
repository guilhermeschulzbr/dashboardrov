# -*- coding: utf-8 -*-
"""Módulo de análise de motoristas"""


import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from config.settings import Config
from utils.formatters import BrazilianFormatter
from components.charts import ChartComponent


class DriverAnalysis:
    """Análise de motoristas"""
    
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config
        self.formatter = BrazilianFormatter()
        self.driver_col = self._find_driver_column()
    
    def render(self):
        """Renderiza análise de motoristas"""
        st.header("👥 Análise de Motoristas")
        
        if not self.driver_col:
            st.warning("Coluna de motorista não encontrada nos dados")
            return
        
        # Tabs de análise
        tab1, tab2, tab3 = st.tabs(["📊 Visão Geral", "🏆 Rankings", "🔍 Detalhes"])
        
        with tab1:
            self.render_overview()
        
        with tab2:
            self.render_rankings()
        
        with tab3:
            self.render_driver_details()
    
    def render_overview(self):
        """Renderiza visão geral dos motoristas"""
        # Calcula KPIs
        kpis = self.calculate_driver_kpis()
        
        # Renderiza KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "👤 Total de Motoristas",
                self.formatter.format_integer(kpis["total_motoristas"])
            )
        
        with col2:
            st.metric(
                "📊 Média Viagens/Motorista",
                self.formatter.format_number(kpis["media_viagens"], 1)
            )
        
        with col3:
            st.metric(
                "👥 Média Passageiros/Motorista",
                self.formatter.format_integer(kpis["media_passageiros"])
            )
        
        with col4:
            st.metric(
                "🛣️ Média KM/Motorista",
                self.formatter.format_number(kpis["media_km"], 1)
            )
        
        # Gráfico de distribuição
        self.render_distribution_chart()
    
    def calculate_driver_kpis(self) -> Dict[str, Any]:
        """Calcula KPIs dos motoristas"""
        kpis = {
            "total_motoristas": 0,
            "media_viagens": 0,
            "media_passageiros": 0,
            "media_km": 0
        }
        
        if not self.driver_col:
            return kpis
        
        # Total de motoristas únicos
        kpis["total_motoristas"] = self.df[self.driver_col].nunique()
        
        if kpis["total_motoristas"] == 0:
            return kpis
        
        # Agregações por motorista
        driver_stats = self.df.groupby(self.driver_col).agg({
            self.driver_col: 'size',  # Contagem de viagens
            **({
                'Passageiros': 'sum',
                'Distancia': 'sum'
            } if all(col in self.df.columns for col in ['Passageiros', 'Distancia']) else {})
        })
        
        # Médias
        kpis["media_viagens"] = len(self.df) / kpis["total_motoristas"]
        
        if 'Passageiros' in driver_stats.columns:
            kpis["media_passageiros"] = driver_stats['Passageiros'].mean()
        
        if 'Distancia' in driver_stats.columns:
            kpis["media_km"] = driver_stats['Distancia'].mean()
        
        return kpis
    
    def render_rankings(self):
        """Renderiza rankings de motoristas"""
        df = self.df.copy()
        # Converte numéricos básicos
        for c in ["Passageiros", "Distancia"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        
        gb = df.groupby(self.driver_col, dropna=False)
        
        # Viagens via size (evita colisão no reset_index)
        driver_data = gb.size().rename("Viagens").reset_index()
        
        # Somatórios opcionais
        if "Passageiros" in df.columns:
            sum_pass = gb["Passageiros"].sum().reset_index().rename(columns={"Passageiros": "Passageiros"})
            driver_data = driver_data.merge(sum_pass, on=self.driver_col, how="left")
        if "Distancia" in df.columns:
            sum_km = gb["Distancia"].sum().reset_index().rename(columns={"Distancia": "Distancia"})
            driver_data = driver_data.merge(sum_km, on=self.driver_col, how="left")
        
        # Renomeia coluna-chave para 'Motorista'
        driver_data = driver_data.rename(columns={self.driver_col: "Motorista"})
        
        # Rankings
        col1, col2 = st.columns(2)
        
        with col1:
            if "Passageiros" in driver_data.columns:
                st.subheader("🏆 Top 10 - Passageiros Transportados")
                top_pass = driver_data.nlargest(10, "Passageiros")[["Motorista", "Passageiros"]].copy()
                top_pass["Passageiros"] = top_pass["Passageiros"].apply(self.formatter.format_integer)
                st.dataframe(top_pass, hide_index=True)
        
        with col2:
            st.subheader("🚀 Top 10 - Mais Viagens")
            top_trips = driver_data.nlargest(10, "Viagens")[["Motorista", "Viagens"]].copy()
            top_trips["Viagens"] = top_trips["Viagens"].apply(self.formatter.format_integer)
            st.dataframe(top_trips, hide_index=True)
        
        # Gráfico de barras dos top motoristas
        if "Passageiros" in driver_data.columns:
            import plotly.express as px
            top_20 = driver_data.nlargest(20, "Passageiros")
            fig = px.bar(
                top_20,
                x="Motorista",
                y="Passageiros",
                title="Top 20 Motoristas por Passageiros",
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)

    def render_driver_details(self):
        """Renderiza detalhes de motorista específico"""
        # Seletor de motorista
        drivers = sorted(self.df[self.driver_col].dropna().unique().tolist())
        
        selected_driver = st.selectbox(
            "Selecione um motorista para análise detalhada:",
            options=[""] + drivers
        )
        
        if not selected_driver:
            st.info("Selecione um motorista para ver os detalhes")
            return
        
        # Filtra dados do motorista
        driver_df = self.df[self.df[self.driver_col] == selected_driver]
        
        # KPIs do motorista
        st.subheader(f"📊 Indicadores - {selected_driver}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Viagens", self.formatter.format_integer(len(driver_df)))
        
        with col2:
            if 'Passageiros' in driver_df.columns:
                st.metric(
                    "Passageiros",
                    self.formatter.format_integer(driver_df['Passageiros'].sum())
                )
        
        with col3:
            if 'Distancia' in driver_df.columns:
                st.metric(
                    "KM Total",
                    self.formatter.format_number(driver_df['Distancia'].sum(), 1)
                )
        
        with col4:
            if all(col in driver_df.columns for col in ['Data Hora Inicio Operacao', 'Data Hora Final Operacao']):
                duracao = (
                    pd.to_datetime(driver_df['Data Hora Final Operacao'], errors='coerce') -
                    pd.to_datetime(driver_df['Data Hora Inicio Operacao'], errors='coerce')
                ).dt.total_seconds() / 3600
                st.metric(
                    "Horas Trabalhadas",
                    self.formatter.format_number(duracao.sum(), 1) + " h"
                )
        
        # Evolução temporal
        if 'Data' in driver_df.columns and 'Passageiros' in driver_df.columns:
            st.subheader("📈 Evolução Temporal")
            
            daily_data = driver_df.groupby('Data')['Passageiros'].sum().reset_index()
            
            import plotly.express as px
            fig = px.line(
                daily_data,
                x='Data',
                y='Passageiros',
                title='Passageiros Transportados por Dia',
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de viagens recentes
        st.subheader("📋 Últimas Viagens")
        
        display_cols = [
            col for col in ['Data', 'Nome Linha', 'Numero Veiculo', 'Passageiros', 'Distancia']
            if col in driver_df.columns
        ]
        
        if display_cols:
            recent_trips = driver_df[display_cols].head(20)
            st.dataframe(recent_trips, hide_index=True)
    
    def render_distribution_chart(self):
        """Renderiza gráfico de distribuição de viagens por motorista"""
        if not self.driver_col:
            return
        
        # Conta viagens por motorista
        trip_counts = self.df[self.driver_col].value_counts()
        
        # Cria histograma
        import plotly.express as px
        
        fig = px.histogram(
            x=trip_counts.values,
            nbins=30,
            title="Distribuição de Viagens por Motorista",
            labels={'x': 'Número de Viagens', 'y': 'Quantidade de Motoristas'}
        )
        
        fig.update_layout(height=300)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _find_driver_column(self) -> Optional[str]:
        """Encontra a coluna de motorista"""
        for col in self.config.COLS_MOTORISTA:
            if col in self.df.columns:
                return col
        return None
