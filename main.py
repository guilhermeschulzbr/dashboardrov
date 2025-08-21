# -*- coding: utf-8 -*-
"""
Dashboard Operacional ROV - Versão Refatorada
Execução: streamlit run main.py
"""


import streamlit as st
from config.settings import Config, PageConfig
from data.loader import DataLoader
from data.processors import DataProcessor
from components.filters import FilterComponent
from components.kpis import KPIComponent
from components.charts import ChartComponent
from components.tables import TableComponent
from modules.fleet import FleetAnalysis
from modules.drivers import DriverAnalysis
from modules.anomalies import AnomalyDetection
from modules.timeline import TimelineAnalysis
from modules.journey_filter import filter_by_daily_journey
from modules.suspicions import SuspicionAnalysis
from utils.cache import CacheManager
from modules.predictions import PredictionAnalysis


# Configuração da página
PageConfig.setup()


class DashboardROV:
    """Classe principal do Dashboard ROV"""
    
    def __init__(self):
        self.config = Config()
        self.cache = CacheManager()
        self.init_session_state()
    
    def init_session_state(self):
        """Inicializa o estado da sessão"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = None
        if 'configs' not in st.session_state:
            st.session_state.configs = {}
    
    def run(self):
        """Executa o dashboard"""
        # Título e configuração inicial
        st.title("📊 Dashboard Operacional ROV")
        st.caption("*Sistema de Análise Operacional de Transporte*")
        
        # Sidebar para upload e configurações
        with st.sidebar:
            self.render_sidebar()
        
        # Verifica se há dados carregados
        if st.session_state.data is None:
            st.info("👆 Por favor, faça upload do arquivo CSV na barra lateral")
            return
        
        # Renderiza o dashboard principal
        self.render_dashboard()
    
    
    def render_dashboard(self):
        """Renderiza o dashboard principal com abas e filtros aplicados."""
        # Determina o DF base com segurança
        df_candidate = st.session_state.get('filtered_data', None)
        if df_candidate is not None and getattr(df_candidate, "empty", True) is False:
            df = df_candidate
        else:
            df = st.session_state.get('data', None)

        # Aplica filtro de jornadas diárias > 09:00h conforme seleção da sidebar
        _mode = st.session_state.get('journey_filter_mode', 'Não filtrar')
        if df is not None and getattr(df, 'empty', True) is False and _mode != 'Não filtrar':
            try:
                df = filter_by_daily_journey(df, mode=_mode, threshold_hours=9.0)
            except Exception as _e:
                st.warning(f'Falha ao aplicar filtro de jornada diária: {_e}')

        if df is None or getattr(df, "empty", True):
            st.info("Sem dados após aplicação dos filtros.")
            return

        # Abas do dashboard
        tabs = st.tabs([
            "📌 Visão Geral",
            "🧩 Frota e Linhas",
            "👥 Motoristas",
            "📈 Análises",
            "⚠️ Alertas",
            "📅 Linha do Tempo"
        ])
        
        with tabs[0]:
            self.render_overview_tab(df)
        with tabs[1]:
            self.render_fleet_tab(df)
        with tabs[2]:
            self.render_drivers_tab(df)
        with tabs[3]:
            self.render_analysis_tab(df)
        with tabs[4]:
            self.render_alerts_tab(df) if hasattr(self, "render_alerts_tab") else st.info("Sem módulo de alertas.")
        with tabs[5]:
            self.render_timeline_tab(df)
    def render_sidebar(self):
            """Renderiza a barra lateral"""
            st.title("⚙️ Configurações")
        
            # Upload de arquivo
            uploaded_file = st.file_uploader(
                "Carregue o arquivo de dados (CSV ';')",
                type=["csv"]
            )
        
            if uploaded_file:
                with st.spinner("Carregando dados..."):
                    loader = DataLoader()
                    df = loader.load_csv(uploaded_file)
                    if df is not None:
                        processor = DataProcessor()
                        df = processor.process_dataframe(df)
                        st.session_state.data = df
                        st.success("✅ Dados carregados com sucesso!")
        
            # Filtros principais
            if st.session_state.data is not None:
                filter_component = FilterComponent(st.session_state.data)
                st.session_state.filtered_data = filter_component.render()

                # Filtro de jornada diária > 09:00h
                st.markdown("### ⏱️ Filtro de Jornada Diária (> 13:00h)")
                st.selectbox(
                    "Aplicar filtro de jornadas por motorista no dia",
                    ["Não filtrar", "Mostrar apenas", "Expurgar"],
                    index=0,
                    key="journey_filter_mode"
                )
    def render_overview_tab(self, df):
        """Renderiza a aba de visão geral"""
        st.header("Visão Geral do Sistema")
        
        # KPIs principais
        kpi_component = KPIComponent(df, self.config)
        kpi_component.render_main_kpis()
        
        # Gráficos principais
        chart_component = ChartComponent(df)
        
        col1, col2 = st.columns(2)
        with col1:
            chart_component.render_daily_passengers()
        with col2:
            chart_component.render_line_ranking()
        
        # Tabela consolidada
        st.subheader("📋 Resumo por Linha")
        table_component = TableComponent(df, self.config)
        table_component.render_consolidated_table()
    
    def render_fleet_tab(self, df):
        """Renderiza a aba de análise de frota"""
        fleet_analysis = FleetAnalysis(df, self.config)
        fleet_analysis.render()
    
    def render_drivers_tab(self, df):
        """Renderiza a aba de análise de motoristas"""
        driver_analysis = DriverAnalysis(df, self.config)
        driver_analysis.render()
    
    def render_analysis_tab(self, df):
        """Renderiza a aba de análises avançadas"""
        st.header("Análises Avançadas")
    
        analysis_type = st.selectbox(
        "Selecione o tipo de análise",
        ["Detecção de Anomalias", "Análise de Suspeições", "Previsões"]
        )
    
        if analysis_type == "Detecção de Anomalias":
            anomaly_detector = AnomalyDetection(df)
            anomaly_detector.render()
        elif analysis_type == "Análise de Suspeições":
            suspicion_analysis = SuspicionAnalysis(df)
            suspicion_analysis.render()
        else:  # Previsões
            prediction_analysis = PredictionAnalysis(df)
            prediction_analysis.render()
    
    def render_alerts_tab(self, df):
        """Renderiza a aba de alertas"""
        st.header("⚠️ Alertas Operacionais")
        
        # Implementar lógica de alertas
        alerts = self.generate_alerts(df)
        
        if alerts:
            for alert_type, alert_df in alerts:
                with st.expander(f"🔴 {alert_type} ({len(alert_df)} registros)"):
                    st.dataframe(alert_df.head(100))
        else:
            st.success("✅ Nenhum alerta identificado")
    
    def render_timeline_tab(self, df):
        """Renderiza a aba de linha do tempo"""
        timeline_analysis = TimelineAnalysis(df)
        timeline_analysis.render()
    
    def generate_alerts(self, df):
        """Gera alertas baseados nos dados"""
        alerts = []
        
        # Viagens sem passageiros
        if "Passageiros" in df.columns:
            zero_pax = df[df["Passageiros"].fillna(0) <= 0]
            if not zero_pax.empty:
                alerts.append(("Viagens sem passageiros", zero_pax))
        
        # Adicionar mais lógicas de alerta conforme necessário
        
        return alerts


if __name__ == "__main__":
    app = DashboardROV()
    app.run()
