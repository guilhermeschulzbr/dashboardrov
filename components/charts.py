# -*- coding: utf-8 -*-
"""Componente de gráficos"""


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any
from config.settings import Config
from utils.formatters import BrazilianFormatter


class ChartComponent:
    """Componente para renderização de gráficos"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.config = Config()
        self.formatter = BrazilianFormatter()
    
    def render_daily_passengers(self):
        """Renderiza gráfico de passageiros por dia"""
        if not {"Data", "Passageiros"}.issubset(self.df.columns):
            st.info("Dados insuficientes para gráfico de passageiros diários")
            return
        
        if not self.df["Data"].notna().any():
            st.info("Sem dados de data válidos")
            return
        
        # Agrupa por data
        daily_data = (
            self.df.groupby("Data", as_index=False)["Passageiros"]
            .sum()
            .sort_values("Data")
        )
        
        # Cria o gráfico
        fig = px.line(
            daily_data,
            x="Data",
            y="Passageiros",
            title="📈 Evolução de Passageiros por Dia",
            markers=True
        )
        
        # Customização
        fig.update_traces(
            line_color=self.config.COLOR_SCHEME["primary"],
            marker_color=self.config.COLOR_SCHEME["info"]
        )
        
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Total de Passageiros",
            xaxis_tickformat="%d/%m/%Y",
            height=self.config.CHART_HEIGHT,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_line_ranking(self, top_n: int = 15):
        """Renderiza ranking de linhas por demanda"""
        if not {"Nome Linha", "Passageiros"}.issubset(self.df.columns):
            st.info("Dados insuficientes para ranking de linhas")
            return
        
        # Agrupa por linha
        line_data = (
            self.df.groupby("Nome Linha", as_index=False)["Passageiros"]
            .sum()
            .sort_values("Passageiros", ascending=False)
            .head(top_n)
        )
        
        # Cria o gráfico
        fig = px.bar(
            line_data,
            x="Nome Linha",
            y="Passageiros",
            title=f"🏆 Top {top_n} Linhas por Passageiros",
            text="Passageiros"
        )
        
        # Customização
        fig.update_traces(
            marker_color=self.config.COLOR_SCHEME["success"],
            texttemplate='%{text:,.0f}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title="Linha",
            yaxis_title="Total de Passageiros",
            xaxis_tickangle=-45,
            height=self.config.CHART_HEIGHT,
            margin=dict(l=10, r=10, t=40, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_heatmap(self):
        """Renderiza heatmap de demanda por hora e dia da semana"""
        if not {"Hora_Base", "DiaSemana_Base", "Passageiros"}.issubset(self.df.columns):
            st.info("Dados insuficientes para heatmap")
            return
        
        # Prepara dados
        heatmap_data = (
            self.df.dropna(subset=["Hora_Base", "DiaSemana_Base"])
            .groupby(["DiaSemana_Base", "Hora_Base"], as_index=False)["Passageiros"]
            .sum()
        )
        
        if heatmap_data.empty:
            st.info("Sem dados para gerar o heatmap")
            return
        
        # Mapeia dias da semana
        day_map = {
            0: "Segunda",
            1: "Terça",
            2: "Quarta",
            3: "Quinta",
            4: "Sexta",
            5: "Sábado",
            6: "Domingo"
        }
        heatmap_data["Dia"] = heatmap_data["DiaSemana_Base"].map(day_map)
        
        # Cria o heatmap
        fig = px.density_heatmap(
            heatmap_data,
            x="Hora_Base",
            y="Dia",
            z="Passageiros",
            title="🔥 Heatmap de Demanda (Hora x Dia da Semana)",
            color_continuous_scale="YlOrRd"
        )
        
        fig.update_layout(
            xaxis_title="Hora do Dia",
            yaxis_title="Dia da Semana",
            height=self.config.CHART_HEIGHT,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_pie_chart(self, 
                        column: str,
                        title: str,
                        top_n: Optional[int] = None):
        """
        Renderiza gráfico de pizza genérico
        
        Args:
            column: Coluna para agregação
            title: Título do gráfico
            top_n: Número de categorias top (None para todas)
        """
        if column not in self.df.columns:
            st.info(f"Coluna '{column}' não encontrada")
            return
        
        # Conta valores
        value_counts = self.df[column].value_counts()
        
        if top_n:
            value_counts = value_counts.head(top_n)
        
        # Cria o gráfico
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=title,
            hole=0.4
        )
        
        fig.update_layout(
            height=self.config.CHART_HEIGHT,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_scatter_plot(self,
                           x_column: str,
                           y_column: str,
                           title: str,
                           color_column: Optional[str] = None,
                           trendline: bool = False):
        """
        Renderiza gráfico de dispersão
        
        Args:
            x_column: Coluna do eixo X
            y_column: Coluna do eixo Y
            title: Título do gráfico
            color_column: Coluna para colorir pontos
            trendline: Se deve adicionar linha de tendência
        """
        required_cols = {x_column, y_column}
        if not required_cols.issubset(self.df.columns):
            st.info(f"Colunas necessárias não encontradas: {required_cols}")
            return
        
        # Prepara argumentos
        plot_args = {
            "data_frame": self.df,
            "x": x_column,
            "y": y_column,
            "title": title
        }
        
        if color_column and color_column in self.df.columns:
            plot_args["color"] = color_column
        
        if trendline:
            plot_args["trendline"] = "ols"
        
        # Cria o gráfico
        fig = px.scatter(**plot_args)
        
        fig.update_layout(
            height=self.config.CHART_HEIGHT,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)