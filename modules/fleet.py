# -*- coding: utf-8 -*-
"""Módulo de análise de frota"""


import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from config.settings import Config
from utils.formatters import BrazilianFormatter
from components.charts import ChartComponent
from components.tables import TableComponent


class FleetAnalysis:
    """Análise de aproveitamento da frota"""
    
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config
        self.formatter = BrazilianFormatter()
    
    def render(self):
        """Renderiza análise de frota"""
        st.header("🚚 Análise de Frota")
        
        # Calcula KPIs
        kpis = self.calculate_fleet_kpis()
        
        # Renderiza KPIs
        self.render_fleet_kpis(kpis)
        
        # Análise por linha
        st.subheader("📊 Aproveitamento por Linha")
        line_analysis = self.analyze_by_line()
        if not line_analysis.empty:
            self.render_line_analysis(line_analysis)
        
        # Análise temporal
        st.subheader("📈 Evolução Temporal")
        self.render_temporal_analysis()
    
    def calculate_fleet_kpis(self) -> Dict[str, Any]:
        """Calcula KPIs da frota"""
        kpis = {}
        
        # Verifica colunas necessárias
        start_col = self._find_column(self.config.COLS_HORARIO_INICIO)
        end_col = self._find_column(self.config.COLS_HORARIO_FIM)
        vehicle_col = self._find_column(self.config.COLS_VEICULO)
        
        if not all([start_col, end_col, vehicle_col]):
            return kpis
        
        # Calcula duração das operações
        self.df["_duracao_horas"] = (
            (pd.to_datetime(self.df[end_col], errors='coerce') -
             pd.to_datetime(self.df[start_col], errors='coerce'))
            .dt.total_seconds() / 3600.0
        )
        
        # Total de horas operacionais
        kpis["horas_totais"] = self.df["_duracao_horas"].sum()
        
        # Dias ativos
        if "Data" in self.df.columns:
            kpis["dias_ativos"] = self.df["Data"].nunique()
        else:
            kpis["dias_ativos"] = 0
        
        # Veículos médios em operação por dia
        if kpis["dias_ativos"] > 0:
            daily_vehicles = self.df.groupby("Data")[vehicle_col].nunique()
            kpis["veiculos_med_operacao"] = daily_vehicles.mean()
        else:
            kpis["veiculos_med_operacao"] = 0
        
        # Horas médias por veículo
        vehicle_hours = self.df.groupby(vehicle_col)["_duracao_horas"].sum()
        kpis["horas_med_veiculo"] = vehicle_hours.mean() if not vehicle_hours.empty else 0
        
        # Taxa de utilização (baseado em 7h20 por dia)
        if kpis["veiculos_med_operacao"] > 0 and kpis["dias_ativos"] > 0:
            horas_disponiveis = kpis["veiculos_med_operacao"] * kpis["dias_ativos"] * (self.config.JORNADA_TOTAL_MIN / 60)
            kpis["taxa_utilizacao"] = kpis["horas_totais"] / horas_disponiveis if horas_disponiveis > 0 else 0
        else:
            kpis["taxa_utilizacao"] = 0
        
        return kpis
    
    def render_fleet_kpis(self, kpis: Dict[str, Any]):
        """Renderiza KPIs da frota"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "⏱️ Horas Totais",
                self.formatter.format_number(kpis.get("horas_totais", 0), 1) + " h"
            )
        
        with col2:
            st.metric(
                "📅 Dias Ativos",
                self.formatter.format_integer(kpis.get("dias_ativos", 0))
            )
        
        with col3:
            st.metric(
                "🚌 Veículos Médios/Dia",
                self.formatter.format_number(kpis.get("veiculos_med_operacao", 0), 1)
            )
        
        with col4:
            taxa = kpis.get("taxa_utilizacao", 0)
            badge = "🟢" if taxa >= 0.9 else ("🟡" if taxa >= 0.75 else "🔴")
            st.metric(
                f"{badge} Taxa de Utilização",
                self.formatter.format_percentage(taxa)
            )
    
    def analyze_by_line(self) -> pd.DataFrame:
        """Analisa aproveitamento por linha"""
        line_col = self._find_column(self.config.COLS_LINHA)
        vehicle_col = self._find_column(self.config.COLS_VEICULO)
        
        if not line_col or not vehicle_col:
            return pd.DataFrame()
        
        if "_duracao_horas" not in self.df.columns:
            return pd.DataFrame()
        
        # Agrega por linha
        agg_dict = {
            "_duracao_horas": "sum",
            vehicle_col: "nunique",
            "Data": "nunique" if "Data" in self.df.columns else lambda x: 0
        }
        
        line_analysis = self.df.groupby(line_col).agg(agg_dict).reset_index()
        line_analysis.columns = ["Linha", "Horas Totais", "Veículos Únicos", "Dias Ativos"]
        
        # Calcula métricas derivadas
        line_analysis["Horas/Dia"] = (
            line_analysis["Horas Totais"] / line_analysis["Dias Ativos"]
        ).replace([np.inf, -np.inf], 0)
        
        line_analysis["Horas/Veículo"] = (
            line_analysis["Horas Totais"] / line_analysis["Veículos Únicos"]
        ).replace([np.inf, -np.inf], 0)
        
        return line_analysis.sort_values("Horas Totais", ascending=False)
    
    def render_line_analysis(self, data: pd.DataFrame):
        """Renderiza análise por linha"""
        # Formata dados para exibição
        display_data = data.copy()
        display_data["Horas Totais"] = display_data["Horas Totais"].apply(
            lambda x: self.formatter.format_number(x, 1) + " h"
        )
        display_data["Horas/Dia"] = display_data["Horas/Dia"].apply(
            lambda x: self.formatter.format_number(x, 1) + " h"
        )
        display_data["Horas/Veículo"] = display_data["Horas/Veículo"].apply(
            lambda x: self.formatter.format_number(x, 1) + " h"
        )
        
        st.dataframe(
            display_data.head(20),
            use_container_width=True,
            hide_index=True
        )
    
    def render_temporal_analysis(self):
        """Renderiza análise temporal da frota"""
        if "Data" not in self.df.columns or "_duracao_horas" not in self.df.columns:
            st.info("Dados insuficientes para análise temporal")
            return
        
        vehicle_col = self._find_column(self.config.COLS_VEICULO)
        if not vehicle_col:
            return
        
        # Agrega por dia
        daily_data = self.df.groupby("Data").agg({
            "_duracao_horas": "sum",
            vehicle_col: "nunique"
        }).reset_index()
        daily_data.columns = ["Data", "Horas Totais", "Veículos"]
        
        # Cria gráfico
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Horas Operacionais", "Veículos em Operação"),
            shared_xaxes=True
        )
        
        # Gráfico de horas
        fig.add_trace(
            go.Scatter(
                x=daily_data["Data"],
                y=daily_data["Horas Totais"],
                mode='lines+markers',
                name='Horas',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Gráfico de veículos
        fig.add_trace(
            go.Bar(
                x=daily_data["Data"],
                y=daily_data["Veículos"],
                name='Veículos',
                marker_color='green'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Data", row=2, col=1)
        fig.update_yaxes(title_text="Horas", row=1, col=1)
        fig.update_yaxes(title_text="Veículos", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _find_column(self, candidates: list) -> str:
        """Encontra primeira coluna existente de uma lista"""
        for col in candidates:
            if col in self.df.columns:
                return col
        return None
