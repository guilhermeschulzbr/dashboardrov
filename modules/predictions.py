# -*- coding: utf-8 -*-
"""Módulo de previsões e análise preditiva"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


from config.settings import Config
from utils.formatters import BrazilianFormatter




class PredictionAnalysis:
    """Análise preditiva e previsões"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.config = Config()
        self.formatter = BrazilianFormatter()
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
    
    def render(self):
        """Renderiza análise de previsões"""
        st.header("🔮 Análise Preditiva e Previsões")
        
        # Verifica se há dados suficientes
        if len(self.df) < 30:
            st.warning("⚠️ São necessários pelo menos 30 dias de dados para realizar previsões confiáveis.")
            return
        
        # Tabs de análise
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Previsão de Demanda",
            "🚌 Previsão de Frota",
            "📊 Análise de Tendências",
            "🎯 Cenários e Simulações"
        ])
        
        with tab1:
            self.render_demand_prediction()
        
        with tab2:
            self.render_fleet_prediction()
        
        with tab3:
            self.render_trend_analysis()
        
        with tab4:
            self.render_scenario_simulation()
    
    def prepare_time_series_data(self, metric: str, aggregation: str = 'daily') -> pd.DataFrame:
        """Prepara dados para análise de séries temporais"""
        df_copy = self.df.copy()
        
        # Garante que temos uma coluna de data
        if 'Data' not in df_copy.columns:
            if 'Data Hora Inicio Operacao' in df_copy.columns:
                df_copy['Data'] = pd.to_datetime(df_copy['Data Hora Inicio Operacao']).dt.date
            else:
                return pd.DataFrame()
        
        # Converte Data para datetime
        df_copy['Data'] = pd.to_datetime(df_copy['Data'])
        
        # Mapeia colunas disponíveis
        available_columns = df_copy.columns.tolist()
        
        # Mapeamento flexível de colunas
        column_mapping = {
            'passageiros': ['Passageiros', 'passageiros', 'Total_Passageiros', 'total_passageiros'],
            'distancia': ['Distancia', 'distancia', 'KM', 'km', 'Quilometragem', 'quilometragem'],
            'veiculo': ['Numero Veiculo', 'numero_veiculo', 'Veiculo', 'veiculo', 'ID_Veiculo'],
            'linha': ['Nome Linha', 'nome_linha', 'Linha', 'linha', 'Codigo_Linha']
        }
        
        # Encontra as colunas corretas
        def find_column(column_list):
            for col in column_list:
                if col in available_columns:
                    return col
            return None
        
        passageiros_col = find_column(column_mapping['passageiros'])
        distancia_col = find_column(column_mapping['distancia'])
        veiculo_col = find_column(column_mapping['veiculo'])
        linha_col = find_column(column_mapping['linha'])
        
        # Agrupa por data
        if aggregation == 'daily':
            df_copy['Data'] = df_copy['Data'].dt.date
            
            # Cria dicionário de agregação baseado nas colunas disponíveis
            agg_dict = {}
            
            if passageiros_col:
                agg_dict[passageiros_col] = 'sum'
            if distancia_col:
                agg_dict[distancia_col] = 'sum'
            if veiculo_col:
                agg_dict[veiculo_col] = 'nunique'
            if linha_col:
                agg_dict[linha_col] = 'count'
            
            if not agg_dict:
                # Se não encontrou nenhuma coluna, retorna DataFrame vazio
                return pd.DataFrame()
            
            grouped = df_copy.groupby('Data').agg(agg_dict).reset_index()
            
            # Renomeia colunas para padrão
            new_columns = ['Data']
            if passageiros_col:
                new_columns.append('Passageiros')
            if distancia_col:
                new_columns.append('KM_Total')
            if veiculo_col:
                new_columns.append('Veiculos_Ativos')
            if linha_col:
                new_columns.append('Total_Viagens')
            
            grouped.columns = new_columns
            
            # Converte Data de volta para datetime
            grouped['Data'] = pd.to_datetime(grouped['Data'])
        
        elif aggregation == 'weekly':
            df_copy['Semana'] = df_copy['Data'].dt.to_period('W')
            
            agg_dict = {}
            if passageiros_col:
                agg_dict[passageiros_col] = 'sum'
            if distancia_col:
                agg_dict[distancia_col] = 'sum'
            if veiculo_col:
                agg_dict[veiculo_col] = 'nunique'
            if linha_col:
                agg_dict[linha_col] = 'count'
            
            if not agg_dict:
                return pd.DataFrame()
            
            grouped = df_copy.groupby('Semana').agg(agg_dict).reset_index()
            grouped['Data'] = grouped['Semana'].dt.to_timestamp()
            
            # Seleciona apenas as colunas que existem
            select_columns = ['Data']
            if passageiros_col:
                grouped['Passageiros'] = grouped[passageiros_col]
                select_columns.append('Passageiros')
            if distancia_col:
                grouped['KM_Total'] = grouped[distancia_col]
                select_columns.append('KM_Total')
            if veiculo_col:
                grouped['Veiculos_Ativos'] = grouped[veiculo_col]
                select_columns.append('Veiculos_Ativos')
            if linha_col:
                grouped['Total_Viagens'] = grouped[linha_col]
                select_columns.append('Total_Viagens')
            
            grouped = grouped[select_columns]
        
        else:  # monthly
            df_copy['Mes'] = df_copy['Data'].dt.to_period('M')
            
            agg_dict = {}
            if passageiros_col:
                agg_dict[passageiros_col] = 'sum'
            if distancia_col:
                agg_dict[distancia_col] = 'sum'
            if veiculo_col:
                agg_dict[veiculo_col] = 'nunique'
            if linha_col:
                agg_dict[linha_col] = 'count'
            
            if not agg_dict:
                return pd.DataFrame()
            
            grouped = df_copy.groupby('Mes').agg(agg_dict).reset_index()
            grouped['Data'] = grouped['Mes'].dt.to_timestamp()
            
            # Seleciona apenas as colunas que existem
            select_columns = ['Data']
            if passageiros_col:
                grouped['Passageiros'] = grouped[passageiros_col]
                select_columns.append('Passageiros')
            if distancia_col:
                grouped['KM_Total'] = grouped[distancia_col]
                select_columns.append('KM_Total')
            if veiculo_col:
                grouped['Veiculos_Ativos'] = grouped[veiculo_col]
                select_columns.append('Veiculos_Ativos')
            if linha_col:
                grouped['Total_Viagens'] = grouped[linha_col]
                select_columns.append('Total_Viagens')
            
            grouped = grouped[select_columns]
        
        return grouped.sort_values('Data')
    
    def create_features(self, df: pd.DataFrame, aggregation: str = 'daily') -> pd.DataFrame:
        """Cria features para o modelo de previsão"""
        df = df.copy()
        
        # Garante que Data está em formato datetime
        df['Data'] = pd.to_datetime(df['Data'])
        
        # Features temporais
        df['dia_semana'] = df['Data'].dt.dayofweek
        df['dia_mes'] = df['Data'].dt.day
        df['mes'] = df['Data'].dt.month
        df['trimestre'] = df['Data'].dt.quarter
        df['dia_ano'] = df['Data'].dt.dayofyear
        df['semana_ano'] = df['Data'].dt.isocalendar().week
        
        # Features de tendência
        min_date = df['Data'].min()
        df['dias_desde_inicio'] = (df['Data'] - min_date).dt.days
        
        # Define lags apropriados baseado na agregação
        if aggregation == 'weekly':
            lags = [1, 2, 4, 8]  # 1, 2, 4 e 8 semanas atrás
            windows = [2, 4, 8]   # Médias de 2, 4 e 8 semanas
        elif aggregation == 'monthly':
            lags = [1, 2, 3, 6]  # 1, 2, 3 e 6 meses atrás
            windows = [2, 3, 6]   # Médias de 2, 3 e 6 meses
        else:  # daily
            lags = [1, 7, 14, 30]  # 1, 7, 14 e 30 dias atrás
            windows = [7, 14, 30]   # Médias de 7, 14 e 30 dias
        
        # Features de lag (valores anteriores) - apenas para colunas que existem
        available_metric_cols = [col for col in ['Passageiros', 'KM_Total', 'Veiculos_Ativos'] if col in df.columns]
        
        for lag in lags:
            for col in available_metric_cols:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Médias móveis
        for window in windows:
            for col in available_metric_cols:
                df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        
        # Remove linhas com NaN nas features principais
        df = df.dropna(subset=['dias_desde_inicio'])
        
        return df
    
    def train_model(self, df: pd.DataFrame, target: str, model_type: str = 'random_forest', aggregation: str = 'daily') -> Tuple:
        """Treina modelo de previsão"""
        # Verifica se a coluna target existe
        if target not in df.columns:
            return None, None, None, None
        
        # Prepara features com a agregação correta
        df_features = self.create_features(df, aggregation)
        
        # Remove linhas com NaN no target
        df_features = df_features.dropna(subset=[target])
        
        if len(df_features) == 0:
            return None, None, None, None
        
        # Define features para o modelo
        feature_cols = [col for col in df_features.columns
                       if col not in ['Data', target, 'Semana', 'Mes']
                       and not col.startswith('Data')]
        
        # Remove colunas com muitos NaN
        feature_cols = [col for col in feature_cols
                       if df_features[col].notna().sum() > len(df_features) * 0.5]
        
        if len(feature_cols) == 0:
            return None, None, None, None
        
        # Separa features e target
        X = df_features[feature_cols].fillna(0)
        y = df_features[target]
        
        # Verifica se temos dados suficientes
        if len(X) < 10:
            return None, None, None, None
        
        # Divide em treino e teste (80/20)
        split_index = max(1, int(len(X) * 0.8))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Treina modelo
        model = self.models[model_type]
        model.fit(X_train, y_train)
        
        # Faz previsões
        y_pred_train = model.predict(X_train)
        
        # Calcula métricas
        metrics = {
            'mae_train': mean_absolute_error(y_train, y_pred_train),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2_train': r2_score(y_train, y_pred_train)
        }
        
        # Se temos dados de teste, calcula métricas de teste
        if len(X_test) > 0:
            y_pred_test = model.predict(X_test)
            metrics.update({
                'mae_test': mean_absolute_error(y_test, y_pred_test),
                'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'r2_test': r2_score(y_test, y_pred_test)
            })
        
        return model, X, y, metrics
    
    def make_future_predictions(self, model, last_data: pd.DataFrame, periods: int, target: str, aggregation: str = 'daily') -> pd.DataFrame:
        """Faz previsões futuras"""
        predictions = []
        
        # Garante que as datas estão no formato correto
        last_data = last_data.copy()
        last_data['Data'] = pd.to_datetime(last_data['Data'])
        
        last_date = last_data['Data'].max()
        min_date = last_data['Data'].min()
        
        # Define frequência baseada na agregação
        if aggregation == 'weekly':
            freq = 'W'
        elif aggregation == 'monthly':
            freq = 'M'
        else:
            freq = 'D'
        
        # Cria datas futuras
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq=freq)
        
        # Define lags apropriados
        if aggregation == 'weekly':
            lags = [1, 2, 4, 8]
            windows = [2, 4, 8]
        elif aggregation == 'monthly':
            lags = [1, 2, 3, 6]
            windows = [2, 3, 6]
        else:
            lags = [1, 7, 14, 30]
            windows = [7, 14, 30]
        
        # Identifica colunas métricas disponíveis
        available_metric_cols = [col for col in ['Passageiros', 'KM_Total', 'Veiculos_Ativos'] if col in last_data.columns]
        
        # Para cada data futura
        for future_date in future_dates:
            # Cria features para a data futura
            future_row = pd.DataFrame({
                'Data': [future_date],
                'dia_semana': [future_date.dayofweek],
                'dia_mes': [future_date.day],
                'mes': [future_date.month],
                'trimestre': [future_date.quarter],
                'dia_ano': [future_date.dayofyear],
                'semana_ano': [future_date.isocalendar()[1]],
                'dias_desde_inicio': [(future_date - min_date).days]
            })
            
            # Adiciona features de lag baseadas nos dados históricos
            for col in available_metric_cols:
                # Usa médias dos últimos períodos
                for lag in lags:
                    if len(last_data) >= lag:
                        future_row[f'{col}_lag_{lag}'] = last_data[col].iloc[-lag:].mean()
                    else:
                        future_row[f'{col}_lag_{lag}'] = last_data[col].mean()
                
                # Médias móveis
                for window in windows:
                    if len(last_data) >= window:
                        future_row[f'{col}_ma_{window}'] = last_data[col].iloc[-window:].mean()
                    else:
                        future_row[f'{col}_ma_{window}'] = last_data[col].mean()
            
            # Garante que temos todas as features necessárias
            for col in model.feature_names_in_:
                if col not in future_row.columns:
                    future_row[col] = 0
            
            # Faz previsão
            pred = model.predict(future_row[model.feature_names_in_])[0]
            predictions.append({
                'Data': future_date,
                target: pred
            })
        
        return pd.DataFrame(predictions)
    
    def render_demand_prediction(self):
        """Renderiza previsão de demanda de passageiros"""
        st.subheader("📈 Previsão de Demanda de Passageiros")
        
        # Controles
        col1, col2, col3 = st.columns(3)
        
        with col1:
            aggregation = st.selectbox(
                "Agregação temporal:",
                ["Diária", "Semanal", "Mensal"],
                key="pred_agg"
            )
            agg_map = {'Diária': 'daily', 'Semanal': 'weekly', 'Mensal': 'monthly'}
            agg_type = agg_map[aggregation]
        
        with col2:
            model_type = st.selectbox(
                "Modelo de previsão:",
                ["Random Forest", "Regressão Linear"],
                key="pred_model"
            )
            model_map = {'Random Forest': 'random_forest', 'Regressão Linear': 'linear'}
            model_key = model_map[model_type]
        
        with col3:
            # Ajusta o período de previsão baseado na agregação
            if agg_type == 'weekly':
                forecast_periods = st.slider(
                    "Semanas para prever:",
                    min_value=4,
                    max_value=52,
                    value=12,
                    step=4,
                    key="pred_weeks"
                )
            elif agg_type == 'monthly':
                forecast_periods = st.slider(
                    "Meses para prever:",
                    min_value=3,
                    max_value=24,
                    value=6,
                    step=3,
                    key="pred_months"
                )
            else:  # daily
                forecast_periods = st.slider(
                    "Dias para prever:",
                    min_value=7,
                    max_value=90,
                    value=30,
                    step=7,
                    key="pred_days"
                )
        
        # Prepara dados
        ts_data = self.prepare_time_series_data('Passageiros', agg_type)
        
        if ts_data.empty or 'Passageiros' not in ts_data.columns:
            st.warning("Dados insuficientes para análise de passageiros")
            return
        
        # Treina modelo com a agregação correta
        with st.spinner("Treinando modelo de previsão..."):
            model, X, y, metrics = self.train_model(ts_data, 'Passageiros', model_key, agg_type)
        
        if model is None:
            st.error("Não foi possível treinar o modelo. Verifique os dados.")
            return
        
        # Faz previsões futuras
        future_predictions = self.make_future_predictions(model, ts_data, forecast_periods, 'Passageiros', agg_type)
        
        # Visualização
        fig = go.Figure()
        
        # Dados históricos
        fig.add_trace(go.Scatter(
            x=ts_data['Data'],
            y=ts_data['Passageiros'],
            mode='lines+markers',
            name='Histórico',
            line=dict(color='blue', width=2)
        ))
        
        # Previsões
        fig.add_trace(go.Scatter(
            x=future_predictions['Data'],
            y=future_predictions['Passageiros'],
            mode='lines+markers',
            name='Previsão',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Previsão de Demanda - {aggregation}',
            xaxis_title='Data',
            yaxis_title='Número de Passageiros',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas do modelo
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE (Treino)", f"{metrics['mae_train']:.1f}")
        with col2:
            st.metric("RMSE (Treino)", f"{metrics['rmse_train']:.1f}")
        with col3:
            st.metric("R² (Treino)", f"{metrics['r2_train']:.3f}")
        
        # Análise das previsões
        avg_prediction = future_predictions['Passageiros'].mean()
        avg_historical = ts_data['Passageiros'].mean()
        growth_rate = ((avg_prediction / avg_historical) - 1) * 100
        
        if growth_rate > 5:
            st.success(f"📈 Tendência de crescimento: +{growth_rate:.1f}% na demanda")
        elif growth_rate < -5:
            st.warning(f"📉 Tendência de declínio: {growth_rate:.1f}% na demanda")
        else:
            st.info(f"📊 Demanda estável: variação de {growth_rate:.1f}%")
    
    def render_fleet_prediction(self):
        """Renderiza previsão de necessidade de frota"""
        st.subheader("🚌 Previsão de Necessidade de Frota")
        
        # Prepara dados
        ts_data = self.prepare_time_series_data('Veiculos_Ativos', 'daily')
        
        if ts_data.empty or 'Veiculos_Ativos' not in ts_data.columns:
            st.warning("Dados insuficientes para análise de frota")
            return
        
        # Verifica se temos dados de passageiros também
        if 'Passageiros' not in ts_data.columns:
            st.warning("Dados de passageiros não disponíveis para análise de eficiência")
            return
        
        # Calcula métricas de eficiência
        ts_data['Passageiros_por_Veiculo'] = ts_data['Passageiros'] / ts_data['Veiculos_Ativos']
        if 'KM_Total' in ts_data.columns:
            ts_data['KM_por_Veiculo'] = ts_data['KM_Total'] / ts_data['Veiculos_Ativos']
        
        # Controles
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_days = st.slider(
                "Dias para prever:",
                min_value=7,
                max_value=60,
                value=30,
                key="fleet_days"
            )
        
        with col2:
            efficiency_target = st.number_input(
                "Meta de passageiros/veículo:",
                min_value=50,
                max_value=500,
                value=int(ts_data['Passageiros_por_Veiculo'].mean()),
                step=10,
                key="efficiency_target"
            )
        
        # Treina modelo para passageiros
        model_pass, _, _, _ = self.train_model(ts_data, 'Passageiros', 'random_forest')
        
        if model_pass is None:
            st.error("Não foi possível treinar o modelo")
            return
        
        # Faz previsões de passageiros
        future_passengers = self.make_future_predictions(model_pass, ts_data, forecast_days, 'Passageiros')
        
        # Calcula necessidade de frota baseada na eficiência
        future_passengers['Veiculos_Necessarios'] = np.ceil(
            future_passengers['Passageiros'] / efficiency_target
        )
        
        # Visualização
        fig = go.Figure()
        
        # Histórico de veículos
        fig.add_trace(go.Scatter(
            x=ts_data['Data'],
            y=ts_data['Veiculos_Ativos'],
            mode='lines+markers',
            name='Frota Histórica',
            line=dict(color='green', width=2)
        ))
        
        # Previsão de necessidade
        fig.add_trace(go.Scatter(
            x=future_passengers['Data'],
            y=future_passengers['Veiculos_Necessarios'],
            mode='lines+markers',
            name='Frota Necessária',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        # Linha de frota atual
        current_fleet = ts_data['Veiculos_Ativos'].iloc[-5:].mean()
        fig.add_hline(
            y=current_fleet,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Frota Atual Média: {current_fleet:.0f}"
        )
        
        fig.update_layout(
            title='Previsão de Necessidade de Frota',
            xaxis_title='Data',
            yaxis_title='Número de Veículos',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise de cenários
        st.markdown("### 📊 Análise de Cenários de Frota")
        
        scenarios = pd.DataFrame({
            'Cenário': ['Pessimista', 'Realista', 'Otimista'],
            'Eficiência (pass/veículo)': [
                efficiency_target * 0.8,
                efficiency_target,
                efficiency_target * 1.2
            ]
        })
        
        for i, row in scenarios.iterrows():
            scenarios.loc[i, 'Frota Necessária'] = np.ceil(
                future_passengers['Passageiros'].mean() / row['Eficiência (pass/veículo)']
            )
        
        scenarios['Diferença vs Atual'] = scenarios['Frota Necessária'] - current_fleet
        scenarios['Variação %'] = (scenarios['Diferença vs Atual'] / current_fleet * 100)
        
        # Formata valores
        scenarios['Eficiência (pass/veículo)'] = scenarios['Eficiência (pass/veículo)'].round(0).astype(int)
        scenarios['Frota Necessária'] = scenarios['Frota Necessária'].round(0).astype(int)
        scenarios['Diferença vs Atual'] = scenarios['Diferença vs Atual'].round(0).astype(int)
        scenarios['Variação %'] = scenarios['Variação %'].round(1)
        
        st.dataframe(scenarios, hide_index=True, use_container_width=True)
        
        # Recomendações
        avg_need = future_passengers['Veiculos_Necessarios'].mean()
        
        if avg_need > current_fleet * 1.1:
            st.warning(f"""
            ⚠️ **Atenção**: A previsão indica necessidade de aumento de frota.
            - Frota atual média: {current_fleet:.0f} veículos
            - Necessidade prevista: {avg_need:.0f} veículos
            - Recomenda-se avaliar aquisição de {avg_need - current_fleet:.0f} veículos adicionais
            """)
        elif avg_need < current_fleet * 0.9:
            st.info(f"""
            ℹ️ **Oportunidade de Otimização**: A frota atual pode estar superdimensionada.
            - Frota atual média: {current_fleet:.0f} veículos
            - Necessidade prevista: {avg_need:.0f} veículos
            - Possível redução de {current_fleet - avg_need:.0f} veículos
            """)
        else:
            st.success(f"""
            ✅ **Frota Adequada**: A frota atual está bem dimensionada para a demanda prevista.
            - Frota atual média: {current_fleet:.0f} veículos
            - Necessidade prevista: {avg_need:.0f} veículos
            """)
    
    def render_trend_analysis(self):
        """Renderiza análise de tendências"""
        st.subheader("📊 Análise de Tendências e Padrões")
        
        # Prepara dados
        ts_data = self.prepare_time_series_data('Passageiros', 'daily')
        
        if ts_data.empty:
            st.warning("Dados insuficientes para análise")
            return
        
        if 'Passageiros' not in ts_data.columns:
            st.warning("Dados de passageiros não disponíveis")
            return
        
        # Adiciona informações temporais
        ts_data['Data'] = pd.to_datetime(ts_data['Data'])
        ts_data['Dia_Semana'] = ts_data['Data'].dt.day_name()
        ts_data['Mes'] = ts_data['Data'].dt.month_name()
        ts_data['Dia_Semana_Num'] = ts_data['Data'].dt.dayofweek
        
        # Análise por dia da semana
        st.markdown("### 📅 Padrão Semanal")
        
        weekly_pattern = ts_data.groupby('Dia_Semana_Num').agg({
            'Passageiros': 'mean',
            'KM_Total': 'mean' if 'KM_Total' in ts_data.columns else lambda x: 0,
            'Veiculos_Ativos': 'mean' if 'Veiculos_Ativos' in ts_data.columns else lambda x: 0
        }).reset_index()
        
        dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        weekly_pattern['Dia'] = weekly_pattern['Dia_Semana_Num'].map(lambda x: dias_semana[x])
        
        # Gráfico de padrão semanal
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=weekly_pattern['Dia'],
            y=weekly_pattern['Passageiros'],
            name='Passageiros',
            marker_color='lightblue',
            yaxis='y'
        ))
        
        if 'Veiculos_Ativos' in ts_data.columns:
            fig.add_trace(go.Scatter(
                x=weekly_pattern['Dia'],
                y=weekly_pattern['Veiculos_Ativos'],
                name='Veículos Ativos',
                mode='lines+markers',
                marker_color='red',
                yaxis='y2'
            ))
        
        fig.update_layout(
            title='Padrão de Demanda Semanal',
            xaxis_title='Dia da Semana',
            yaxis=dict(title='Passageiros', side='left'),
            yaxis2=dict(title='Veículos Ativos', overlaying='y', side='right'),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Identificação de dias atípicos
        col1, col2 = st.columns(2)
        
        with col1:
            # Dia com maior demanda
            max_day = weekly_pattern.loc[weekly_pattern['Passageiros'].idxmax()]
            st.info(f"""
            📈 **Dia de Maior Demanda**
            - {max_day['Dia']}
            - Média: {max_day['Passageiros']:.0f} passageiros
            """)
        
        with col2:
            # Dia com menor demanda
            min_day = weekly_pattern.loc[weekly_pattern['Passageiros'].idxmin()]
            st.info(f"""
            📉 **Dia de Menor Demanda**
            - {min_day['Dia']}
            - Média: {min_day['Passageiros']:.0f} passageiros
            """)
        
        # Análise de tendência de longo prazo
        st.markdown("### 📈 Tendência de Longo Prazo")
        
        # Calcula médias móveis
        ts_data['MA7'] = ts_data['Passageiros'].rolling(window=7, min_periods=1).mean()
        ts_data['MA30'] = ts_data['Passageiros'].rolling(window=30, min_periods=1).mean()
        
        # Regressão linear para tendência
        X = np.arange(len(ts_data)).reshape(-1, 1)
        y = ts_data['Passageiros'].values
        lr = LinearRegression()
        lr.fit(X, y)
        trend_line = lr.predict(X)
        
        # Gráfico de tendência
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ts_data['Data'],
            y=ts_data['Passageiros'],
            mode='lines',
            name='Dados Diários',
            line=dict(color='lightgray', width=1),
            opacity=0.5
        ))
        
        fig.add_trace(go.Scatter(
            x=ts_data['Data'],
            y=ts_data['MA7'],
            mode='lines',
            name='Média 7 dias',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=ts_data['Data'],
            y=ts_data['MA30'],
            mode='lines',
            name='Média 30 dias',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=ts_data['Data'],
            y=trend_line,
            mode='lines',
            name='Tendência Linear',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Análise de Tendência de Passageiros',
            xaxis_title='Data',
            yaxis_title='Número de Passageiros',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise da tendência
        trend_direction = "crescente" if lr.coef_[0] > 0 else "decrescente"
        daily_change = lr.coef_[0]
        monthly_change = daily_change * 30
        
        st.info(f"""
        📊 **Análise da Tendência**
        - Tendência: **{trend_direction}**
        - Variação diária média: {daily_change:.1f} passageiros
        - Variação mensal estimada: {monthly_change:.0f} passageiros
        - Taxa de crescimento: {(monthly_change / ts_data['Passageiros'].mean() * 100):.1f}% ao mês
        """)
        
        # Sazonalidade
        st.markdown("### 🌡️ Análise de Sazonalidade")
        
        if len(ts_data) >= 60:  # Precisa de pelo menos 2 meses
            monthly_avg = ts_data.groupby(ts_data['Data'].dt.month)['Passageiros'].mean().reset_index()
            monthly_avg.columns = ['Mês', 'Passageiros']
            monthly_avg['Mês'] = monthly_avg['Mês'].map({
                1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
            })
            
            fig = px.bar(
                monthly_avg,
                x='Mês',
                y='Passageiros',
                title='Média de Passageiros por Mês',
                color='Passageiros',
                color_continuous_scale='RdYlGn'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_scenario_simulation(self):
        """Renderiza simulação de cenários"""
        st.subheader("🎯 Simulação de Cenários")
        
        st.info("""
        Esta seção permite simular diferentes cenários e avaliar seu impacto na operação.
        Ajuste os parâmetros abaixo para ver as projeções.
        """)
        
        # Prepara dados base
        ts_data = self.prepare_time_series_data('Passageiros', 'daily')
        
        if ts_data.empty:
            st.warning("Dados insuficientes para simulação")
            return
        
        if 'Passageiros' not in ts_data.columns:
            st.warning("Dados de passageiros não disponíveis")
            return
        
        # Parâmetros de simulação
        st.markdown("### ⚙️ Parâmetros de Simulação")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            growth_rate = st.slider(
                "Taxa de crescimento anual (%):",
                min_value=-20,
                max_value=50,
                value=10,
                step=5,
                key="growth_rate"
            )
        
        with col2:
            fleet_change = st.slider(
                "Variação da frota (%):",
                min_value=-30,
                max_value=30,
                value=0,
                step=5,
                key="fleet_change"
            )
        
        with col3:
            efficiency_change = st.slider(
                "Melhoria de eficiência (%):",
                min_value=-20,
                max_value=20,
                value=0,
                step=5,
                key="efficiency_change"
            )
        
        # Parâmetros operacionais
        st.markdown("### 🚌 Parâmetros Operacionais")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fuel_price = st.number_input(
                "Preço do combustível (R$/L):",
                min_value=3.0,
                max_value=10.0,
                value=5.5,
                step=0.1,
                format="%.2f",
                key="fuel_price"
            )
        
        with col2:
            avg_consumption = st.number_input(
                "Consumo médio (km/L):",
                min_value=2.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                format="%.1f",
                key="avg_consumption"
            )
        
        with col3:
            ticket_price = st.number_input(
                "Tarifa média (R$):",
                min_value=3.0,
                max_value=10.0,
                value=4.5,
                step=0.5,
                format="%.2f",
                key="ticket_price"
            )
        
        # Horizonte de simulação
        simulation_months = st.slider(
            "Horizonte de simulação (meses):",
            min_value=3,
            max_value=24,
            value=12,
            step=3,
            key="sim_months"
        )
        
        # Executa simulação
        if st.button("🚀 Executar Simulação", type="primary"):
            with st.spinner("Executando simulação..."):
                # Calcula valores base
                base_passengers = ts_data['Passageiros'].mean()
                base_km = ts_data['KM_Total'].mean() if 'KM_Total' in ts_data.columns else base_passengers * 20  # Estimativa
                base_vehicles = ts_data['Veiculos_Ativos'].mean() if 'Veiculos_Ativos' in ts_data.columns else base_passengers / 100  # Estimativa
                
                # Projeta valores futuros
                months = np.arange(simulation_months)
                monthly_growth = (1 + growth_rate/100) ** (1/12)
                
                # Passageiros projetados
                passengers_proj = base_passengers * (monthly_growth ** months)
                
                # Frota projetada
                fleet_proj = base_vehicles * (1 + fleet_change/100)
                
                # KM projetados (ajustado pela eficiência)
                km_proj = base_km * (monthly_growth ** months) * (1 - efficiency_change/100)
                
                # Calcula métricas financeiras
                revenue_proj = passengers_proj * ticket_price * 30  # Mensal
                fuel_cost_proj = (km_proj * 30 / avg_consumption) * fuel_price
                profit_proj = revenue_proj - fuel_cost_proj
                
                # Cria DataFrame de resultados
                simulation_results = pd.DataFrame({
                    'Mês': months + 1,
                    'Passageiros/dia': passengers_proj,
                    'KM/dia': km_proj,
                    'Receita Mensal': revenue_proj,
                    'Custo Combustível': fuel_cost_proj,
                    'Resultado': profit_proj
                })
                
                # Visualização dos resultados
                st.markdown("### 📊 Resultados da Simulação")
                
                # Gráfico de métricas operacionais
                fig1 = go.Figure()
                
                fig1.add_trace(go.Scatter(
                    x=simulation_results['Mês'],
                    y=simulation_results['Passageiros/dia'],
                    mode='lines+markers',
                    name='Passageiros/dia',
                    yaxis='y'
                ))
                
                fig1.add_trace(go.Scatter(
                    x=simulation_results['Mês'],
                    y=simulation_results['KM/dia'],
                    mode='lines+markers',
                    name='KM/dia',
                    yaxis='y2'
                ))
                
                fig1.update_layout(
                    title='Projeção de Métricas Operacionais',
                    xaxis_title='Mês',
                    yaxis=dict(title='Passageiros/dia', side='left'),
                    yaxis2=dict(title='KM/dia', overlaying='y', side='right'),
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                # Gráfico financeiro
                fig2 = go.Figure()
                
                fig2.add_trace(go.Bar(
                    x=simulation_results['Mês'],
                    y=simulation_results['Receita Mensal'],
                    name='Receita',
                    marker_color='green'
                ))
                
                fig2.add_trace(go.Bar(
                    x=simulation_results['Mês'],
                    y=-simulation_results['Custo Combustível'],
                    name='Custo Combustível',
                    marker_color='red'
                ))
                
                fig2.add_trace(go.Scatter(
                    x=simulation_results['Mês'],
                    y=simulation_results['Resultado'],
                    mode='lines+markers',
                    name='Resultado',
                    line=dict(color='blue', width=3)
                ))
                
                fig2.update_layout(
                    title='Projeção Financeira',
                    xaxis_title='Mês',
                    yaxis_title='Valor (R$)',
                    barmode='relative',
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Resumo executivo
                st.markdown("### 📋 Resumo Executivo")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_revenue = simulation_results['Receita Mensal'].sum()
                    st.metric(
                        "Receita Total Projetada",
                        f"R$ {total_revenue:,.0f}",
                        f"{((total_revenue / (base_passengers * ticket_price * 30 * simulation_months)) - 1) * 100:.1f}%"
                    )
                
                with col2:
                    total_cost = simulation_results['Custo Combustível'].sum()
                    st.metric(
                        "Custo Total Combustível",
                        f"R$ {total_cost:,.0f}",
                        f"{((total_cost / (base_km * 30 * simulation_months / avg_consumption * fuel_price)) - 1) * 100:.1f}%"
                    )
                
                with col3:
                    total_profit = simulation_results['Resultado'].sum()
                    margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
                    st.metric(
                        "Resultado Total",
                        f"R$ {total_profit:,.0f}",
                        f"Margem: {margin:.1f}%"
                    )
                
                # Tabela detalhada
                with st.expander("📊 Tabela Detalhada"):
                    # Formata valores para exibição
                    display_df = simulation_results.copy()
                    display_df['Passageiros/dia'] = display_df['Passageiros/dia'].round(0).astype(int)
                    display_df['KM/dia'] = display_df['KM/dia'].round(0).astype(int)
                    display_df['Receita Mensal'] = display_df['Receita Mensal'].apply(lambda x: f"R$ {x:,.0f}")
                    display_df['Custo Combustível'] = display_df['Custo Combustível'].apply(lambda x: f"R$ {x:,.0f}")
                    display_df['Resultado'] = display_df['Resultado'].apply(lambda x: f"R$ {x:,.0f}")
                    
                    st.dataframe(display_df, hide_index=True, use_container_width=True)
                
                # Recomendações baseadas na simulação
                st.markdown("### 💡 Recomendações")
                
                if total_profit > 0:
                    st.success(f"""
                    ✅ **Cenário Positivo**
                    - O cenário simulado projeta resultados positivos
                    - Margem operacional de {margin:.1f}%
                    - Considere implementar as melhorias de eficiência planejadas
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **Cenário de Atenção**
                    - O cenário simulado projeta resultados negativos
                    - Revise os parâmetros de eficiência e custos
                    - Considere ajustar a tarifa ou reduzir custos operacionais
                    """)
                
                # Análise de sensibilidade
                st.markdown("### 🎯 Análise de Sensibilidade")
                
                sensitivity_params = {
                    'Tarifa +10%': ticket_price * 1.1,
                    'Combustível +20%': fuel_price * 1.2,
                    'Eficiência +10%': efficiency_change + 10
                }
                
                sensitivity_results = []
                
                for scenario, value in sensitivity_params.items():
                    if 'Tarifa' in scenario:
                        scenario_revenue = passengers_proj[-1] * value * 30
                    else:
                        scenario_revenue = passengers_proj[-1] * ticket_price * 30
                    
                    if 'Combustível' in scenario:
                        scenario_cost = (km_proj[-1] * 30 / avg_consumption) * value
                    elif 'Eficiência' in scenario:
                        scenario_cost = (km_proj[-1] * 30 * (1 - value/100) / avg_consumption) * fuel_price
                    else:
                        scenario_cost = (km_proj[-1] * 30 / avg_consumption) * fuel_price
                    
                    scenario_profit = scenario_revenue - scenario_cost
                    
                    sensitivity_results.append({
                        'Cenário': scenario,
                        'Receita Mensal': f"R$ {scenario_revenue:,.0f}",
                        'Custo Mensal': f"R$ {scenario_cost:,.0f}",
                        'Resultado': f"R$ {scenario_profit:,.0f}",
                        'Impacto': f"{((scenario_profit - profit_proj[-1]) / profit_proj[-1] * 100):.1f}%"
                    })
                
                sensitivity_df = pd.DataFrame(sensitivity_results)
                st.dataframe(sensitivity_df, hide_index=True, use_container_width=True)
