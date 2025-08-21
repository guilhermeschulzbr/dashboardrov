# -*- coding: utf-8 -*-
"""Módulo de detecção de anomalias"""


import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from config.settings import Config
from utils.formatters import BrazilianFormatter


class AnomalyDetection:
    """Detecção de anomalias nos dados"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.config = Config()
        self.formatter = BrazilianFormatter()
        self.has_sklearn = self._check_sklearn()
    
    def _check_sklearn(self) -> bool:
        """Verifica se scikit-learn está disponível"""
        try:
            from sklearn.ensemble import IsolationForest
            return True
        except ImportError:
            return False
    
    def render(self):
        """Renderiza análise de anomalias"""
        st.header("⚠️ Detecção de Anomalias")
        
        if not self.has_sklearn:
            st.error(
                "O módulo scikit-learn não está instalado. "
                "Execute: pip install scikit-learn"
            )
            return
        
        # Configurações
        col1, col2 = st.columns(2)
        
        with col1:
            contamination = st.slider(
                "Taxa de Contaminação (%)",
                min_value=1,
                max_value=10,
                value=3,
                help="Percentual esperado de anomalias no conjunto"
            ) / 100
        
        with col2:
            min_samples = st.number_input(
                "Mínimo de Amostras",
                min_value=10,
                value=50,
                help="Mínimo de registros para executar a detecção"
            )
        
        # Executa detecção
        anomalies_df, features_used = self.detect_anomalies(contamination, min_samples)
        
        if anomalies_df is None:
            st.info("Dados insuficientes para detecção de anomalias")
            return
        
        # Resultados
        st.subheader("🔍 Resultados da Análise")
        
        # KPIs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total de Registros",
                self.formatter.format_integer(len(self.df))
            )
        
        with col2:
            st.metric(
                "Anomalias Detectadas",
                self.formatter.format_integer(len(anomalies_df))
            )
        
        with col3:
            pct_anomalies = (len(anomalies_df) / len(self.df)) * 100
            st.metric(
                "Percentual de Anomalias",
                self.formatter.format_percentage(pct_anomalies / 100)
            )
        
        # Features usadas
        with st.expander("📊 Features Utilizadas"):
            st.write(features_used)
        
        # Visualização
        if not anomalies_df.empty:
            self.render_anomaly_visualization(anomalies_df)
            
            # Tabela de anomalias
            st.subheader("📋 Registros Anômalos")
            
            display_cols = [
                col for col in ['Data', 'Nome Linha', 'Numero Veiculo', 
                               'Passageiros', 'Distancia', 'anomaly_score']
                if col in anomalies_df.columns
            ]
            
            if display_cols:
                display_df = anomalies_df[display_cols].head(50)
                
                # Formata colunas
                if 'Passageiros' in display_df.columns:
                    display_df['Passageiros'] = display_df['Passageiros'].apply(
                        self.formatter.format_integer
                    )
                
                if 'Distancia' in display_df.columns:
                    display_df['Distancia'] = display_df['Distancia'].apply(
                        lambda x: self.formatter.format_number(x, 1)
                    )
                
                if 'anomaly_score' in display_df.columns:
                    display_df['Score'] = display_df['anomaly_score'].apply(
                        lambda x: self.formatter.format_number(x, 3)
                    )
                    display_df = display_df.drop('anomaly_score', axis=1)
                
                st.dataframe(display_df, hide_index=True)
                
                # Download
                csv = anomalies_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "📥 Baixar Anomalias (CSV)",
                    data=csv,
                    file_name="anomalias_detectadas.csv",
                    mime="text/csv"
                )
    
    def detect_anomalies(self, 
                        contamination: float,
                        min_samples: int) -> Tuple[Optional[pd.DataFrame], list]:
        """
        Detecta anomalias usando Isolation Forest
        
        Returns:
            Tupla (DataFrame com anomalias, lista de features usadas)
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        # Prepara features
        features = []
        feature_names = []
        
        # Passageiros
        if 'Passageiros' in self.df.columns:
            features.append(self.df['Passageiros'].fillna(0))
            feature_names.append('Passageiros')
        
        # Distância
        if 'Distancia' in self.df.columns:
            features.append(self.df['Distancia'].fillna(0))
            feature_names.append('Distância')
        
        # Duração
        if all(col in self.df.columns for col in ['Data Hora Inicio Operacao', 'Data Hora Final Operacao']):
            duracao = (
                pd.to_datetime(self.df['Data Hora Final Operacao'], errors='coerce') -
                pd.to_datetime(self.df['Data Hora Inicio Operacao'], errors='coerce')
            ).dt.total_seconds() / 60
            features.append(duracao.fillna(0))
            feature_names.append('Duração (min)')
        
        # Hora do dia
        if 'Hora_Base' in self.df.columns:
            features.append(self.df['Hora_Base'].fillna(0))
            feature_names.append('Hora')
        
        # Verifica se há features suficientes
        if len(features) < 2:
            return None, []
        
        # Cria matriz de features
        X = pd.DataFrame(dict(zip(feature_names, features)))
        
        # Remove linhas com muitos NaNs
        X = X.dropna(thresh=len(X.columns) * 0.5)
        
        if len(X) < min_samples:
            return None, feature_names
        
        # Normaliza features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Treina modelo
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(X_scaled)
        scores = iso_forest.score_samples(X_scaled)
        
        # Identifica anomalias
        anomaly_mask = predictions == -1
        
        # Cria DataFrame com anomalias
        anomalies_df = self.df.loc[X.index[anomaly_mask]].copy()
        anomalies_df['anomaly_score'] = -scores[anomaly_mask]  # Inverte para maior = mais anômalo
        
        # Ordena por score
        anomalies_df = anomalies_df.sort_values('anomaly_score', ascending=False)
        
        return anomalies_df, feature_names
    
    def render_anomaly_visualization(self, anomalies_df: pd.DataFrame):
        """Renderiza visualização das anomalias"""
        import plotly.express as px
        
        # Scatter plot: Passageiros vs Distância
        if all(col in self.df.columns for col in ['Passageiros', 'Distancia']):
            # Marca anomalias
            plot_df = self.df.copy()
            plot_df['É Anomalia'] = plot_df.index.isin(anomalies_df.index)
            
            fig = px.scatter(
                plot_df,
                x='Distancia',
                y='Passageiros',
                color='É Anomalia',
                title='Distribuição: Passageiros vs Distância',
                color_discrete_map={True: 'red', False: 'blue'},
                opacity=0.6
            )
            
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)