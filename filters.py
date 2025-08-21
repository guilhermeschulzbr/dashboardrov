# -*- coding: utf-8 -*-
"""Componente de filtros"""


import streamlit as st
import pandas as pd
from datetime import date, datetime
from typing import Optional, List
from config.settings import Config


class FilterComponent:
    """Componente para renderização de filtros"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.config = Config()
        self.filtered_df = df.copy()
    
    def render(self) -> pd.DataFrame:
        """
        Renderiza todos os filtros e retorna DataFrame filtrado
        
        Returns:
            DataFrame filtrado
        """
        st.header("🔍 Filtros")
        
        # Filtro de período
        self.apply_date_filter()
        
        # Filtro de dias da semana
        self.apply_weekday_filter()
        
        # Filtro de linhas
        self.apply_line_filter()
        
        # Filtro de veículos
        self.apply_vehicle_filter()
        
        # Filtro de motoristas
        self.apply_driver_filter()
        
        # Filtro de terminal
        self.apply_terminal_filter()
        
        # Opções de expurgo
        self.apply_purge_options()
        
        return self.filtered_df
    
    def apply_date_filter(self):
        """Aplica filtro de período"""
        # Procura por colunas de data disponíveis
        date_columns = []
        for col in self.filtered_df.columns:
            if any(keyword in col.lower() for keyword in ['data', 'date']):
                if self.filtered_df[col].notna().any():
                    date_columns.append(col)
        
        if not date_columns:
            return
        
        # Usa a primeira coluna de data encontrada
        date_col = date_columns[0]
        
        # Converte para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(self.filtered_df[date_col]):
            self.filtered_df[date_col] = pd.to_datetime(self.filtered_df[date_col], errors='coerce')
        
        # Remove valores NaT
        self.filtered_df = self.filtered_df.dropna(subset=[date_col])
        
        if len(self.filtered_df) > 0:
            min_date = self.filtered_df[date_col].min().date()
            max_date = self.filtered_df[date_col].max().date()
            
            date_range = st.date_input(
                "📅 Período",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                format="DD/MM/YYYY"
            )
            
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                mask = (
                    (self.filtered_df[date_col].dt.date >= start_date) &
                    (self.filtered_df[date_col].dt.date <= end_date)
                )
                self.filtered_df = self.filtered_df[mask]
    
    def apply_weekday_filter(self):
        """Aplica filtro de dias da semana"""
        # Procura por colunas de data disponíveis
        date_columns = []
        for col in self.filtered_df.columns:
            if any(keyword in col.lower() for keyword in ['data', 'date']):
                if self.filtered_df[col].notna().any():
                    date_columns.append(col)
        
        if not date_columns:
            return
        
        # Usa a primeira coluna de data encontrada
        date_col = date_columns[0]
        
        # Garante que a coluna está em formato datetime
        if not pd.api.types.is_datetime64_any_dtype(self.filtered_df[date_col]):
            self.filtered_df[date_col] = pd.to_datetime(self.filtered_df[date_col], errors='coerce')
        
        # Remove valores NaT
        self.filtered_df = self.filtered_df.dropna(subset=[date_col])
        
        if len(self.filtered_df) == 0:
            return
        
        # Opções de filtro de dias da semana
        weekday_options = [
            "Todos os dias",
            "Apenas dias úteis (Segunda a Sexta)",
            "Apenas sábados",
            "Apenas domingos",
            "Apenas finais de semana (Sábado e Domingo)",
            "Seleção personalizada"
        ]
        
        selected_option = st.selectbox(
            "📅 Dias da Semana",
            options=weekday_options,
            index=0
        )
        
        if selected_option == "Apenas dias úteis (Segunda a Sexta)":
            # Segunda-feira = 0, Domingo = 6
            # Dias úteis: 0, 1, 2, 3, 4 (Segunda a Sexta)
            self.filtered_df = self.filtered_df[
                self.filtered_df[date_col].dt.dayofweek.isin([0, 1, 2, 3, 4])
            ]
        
        elif selected_option == "Apenas sábados":
            # Sábado = 5
            self.filtered_df = self.filtered_df[
                self.filtered_df[date_col].dt.dayofweek == 5
            ]
        
        elif selected_option == "Apenas domingos":
            # Domingo = 6
            self.filtered_df = self.filtered_df[
                self.filtered_df[date_col].dt.dayofweek == 6
            ]
        
        elif selected_option == "Apenas finais de semana (Sábado e Domingo)":
            # Sábado = 5, Domingo = 6
            self.filtered_df = self.filtered_df[
                self.filtered_df[date_col].dt.dayofweek.isin([5, 6])
            ]
        
        elif selected_option == "Seleção personalizada":
            # Permite seleção individual de dias
            days_mapping = {
                "Segunda-feira": 0,
                "Terça-feira": 1,
                "Quarta-feira": 2,
                "Quinta-feira": 3,
                "Sexta-feira": 4,
                "Sábado": 5,
                "Domingo": 6
            }
            
            selected_days = st.multiselect(
                "Selecione os dias da semana:",
                options=list(days_mapping.keys()),
                default=list(days_mapping.keys())
            )
            
            if selected_days:
                day_numbers = [days_mapping[day] for day in selected_days]
                self.filtered_df = self.filtered_df[
                    self.filtered_df[date_col].dt.dayofweek.isin(day_numbers)
                ]
        
        # Mostra estatísticas dos dias filtrados
        if len(self.filtered_df) > 0 and selected_option != "Todos os dias":
            total_days = len(self.filtered_df[date_col].dt.date.unique())
            st.info(f"📊 {total_days} dias únicos após filtro de dias da semana")
    
    def apply_line_filter(self):
        """Aplica filtro de linhas"""
        if "Nome Linha" in self.filtered_df.columns:
            lines = sorted(self.filtered_df["Nome Linha"].dropna().unique().tolist())
            
            if lines:
                selected_lines = st.multiselect(
                    "🚌 Linhas",
                    options=lines,
                    default=None
                )
                
                if selected_lines:
                    self.filtered_df = self.filtered_df[
                        self.filtered_df["Nome Linha"].isin(selected_lines)
                    ]
    
    def apply_vehicle_filter(self):
        """Aplica filtro de veículos"""
        # Procura por colunas de veículo
        vehicle_columns = []
        for col in self.filtered_df.columns:
            if any(keyword in col.lower() for keyword in ['veiculo', 'vehicle', 'numero']):
                if self.filtered_df[col].notna().any():
                    vehicle_columns.append(col)
        
        if not vehicle_columns:
            return
        
        # Usa a primeira coluna de veículo encontrada
        vehicle_col = vehicle_columns[0]
        
        vehicles = sorted(
            self.filtered_df[vehicle_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        
        if vehicles:
            selected_vehicles = st.multiselect(
                "🚐 Veículos",
                options=vehicles,
                default=None
            )
            
            if selected_vehicles:
                self.filtered_df = self.filtered_df[
                    self.filtered_df[vehicle_col].astype(str).isin(selected_vehicles)
                ]
    
    def apply_driver_filter(self):
        """Aplica filtro de motoristas"""
        if "Cobrador/Operador" in self.filtered_df.columns:
            drivers = sorted(
                self.filtered_df["Cobrador/Operador"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            
            if drivers:
                selected_drivers = st.multiselect(
                    "👤 Motoristas/Operadores",
                    options=drivers,
                    default=None
                )
                
                if selected_drivers:
                    self.filtered_df = self.filtered_df[
                        self.filtered_df["Cobrador/Operador"].astype(str).isin(selected_drivers)
                    ]
    
    def apply_terminal_filter(self):
        """Aplica filtro de terminal"""
        if "Descricao Terminal" in self.filtered_df.columns:
            terminals = sorted(
                self.filtered_df["Descricao Terminal"]
                .dropna()
                .unique()
                .tolist()
            )
            
            if terminals:
                selected_terminals = st.multiselect(
                    "🏢 Terminais",
                    options=terminals,
                    default=None
                )
                
                if selected_terminals:
                    self.filtered_df = self.filtered_df[
                        self.filtered_df["Descricao Terminal"].isin(selected_terminals)
                    ]
    
    def apply_purge_options(self):
        """Aplica opções de expurgo"""
        st.subheader("🗑️ Expurgo de Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            purge_zero = st.checkbox(
                "Expurgar viagens com 0 passageiros",
                value=False
            )
            
            if purge_zero and "Passageiros" in self.filtered_df.columns:
                self.filtered_df = self.filtered_df[
                    self.filtered_df["Passageiros"].fillna(0) > 0
                ]
        
        with col2:
            purge_training = st.checkbox(
                "Expurgar motoristas em treinamento",
                value=False
            )
            
            if purge_training:
                # Implementar lógica de detecção de treinamento
                keywords = ["trein", "treinamento", "teste", "training"]
                pattern = "|".join(keywords)
                
                for col in self.filtered_df.columns:
                    if self.filtered_df[col].dtype == 'object':
                        mask = ~self.filtered_df[col].astype(str).str.contains(
                            pattern,
                            case=False,
                            na=False
                        )
                        self.filtered_df = self.filtered_df[mask]
        
        # Adiciona opção para expurgar dados inconsistentes
        col3, col4 = st.columns(2)
        
        with col3:
            purge_outliers = st.checkbox(
                "Expurgar valores extremos",
                value=False,
                help="Remove registros com valores muito acima ou abaixo da média"
            )
            
            if purge_outliers:
                # Identifica colunas numéricas para análise de outliers
                numeric_columns = self.filtered_df.select_dtypes(include=['number']).columns
                
                for col in numeric_columns:
                    if self.filtered_df[col].notna().sum() > 10:  # Só analisa se tiver dados suficientes
                        Q1 = self.filtered_df[col].quantile(0.25)
                        Q3 = self.filtered_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 3 * IQR  # Usa 3*IQR para ser menos restritivo
                        upper_bound = Q3 + 3 * IQR
                        
                        # Remove outliers extremos
                        self.filtered_df = self.filtered_df[
                            (self.filtered_df[col] >= lower_bound) &
                            (self.filtered_df[col] <= upper_bound)
                        ]
        
        with col4:
            purge_incomplete = st.checkbox(
                "Expurgar registros incompletos",
                value=False,
                help="Remove registros com muitos campos vazios"
            )
            
            if purge_incomplete:
                # Remove registros com mais de 50% de campos vazios
                threshold = len(self.filtered_df.columns) * 0.5
                self.filtered_df = self.filtered_df.dropna(thresh=threshold)
        
        # Mostra estatísticas após filtros
        if len(self.df) != len(self.filtered_df):
            reduction_pct = (1 - len(self.filtered_df) / len(self.df)) * 100
            st.info(f"""
            📊 **Resultado dos Filtros:**
            - Registros originais: {len(self.df):,}
            - Registros após filtros: {len(self.filtered_df):,}
            - Redução: {reduction_pct:.1f}%
            """)
    
    def _find_driver_column(self) -> Optional[str]:
        """Encontra a coluna de motorista - mantido para compatibilidade"""
        # Prioriza a coluna específica do CSV
        if "Cobrador/Operador" in self.filtered_df.columns:
            return "Cobrador/Operador"
        
        # Procura por padrões comuns de colunas de motorista
        driver_patterns = [
            'motorista', 'driver', 'condutor', 'operator', 'nome_motorista',
            'nome motorista', 'codigo_motorista', 'id_motorista'
        ]
        
        for col in self.filtered_df.columns:
            for pattern in driver_patterns:
                if pattern.lower() in col.lower():
                    return col
        
        # Se tiver config, usa as colunas definidas lá
        if hasattr(self.config, 'COLS_MOTORISTA'):
            for col in self.config.COLS_MOTORISTA:
                if col in self.filtered_df.columns:
                    return col
        
        return None
