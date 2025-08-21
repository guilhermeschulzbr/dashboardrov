# -*- coding: utf-8 -*-
"""Módulo de processamento de dados"""


import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from config.settings import Config


class DataProcessor:
    """Classe responsável pelo processamento de dados"""
    
    def __init__(self):
        self.config = Config()
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa o DataFrame aplicando todas as transformações necessárias
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame processado
        """
        df = df.copy()
        
        # Processa datas
        df = self.process_dates(df)
        
        # Processa valores numéricos
        df = self.process_numeric_columns(df)
        
        # Processa categorias
        df = self.process_categorical_columns(df)
        
        # Adiciona colunas derivadas
        df = self.add_derived_columns(df)
        
        return df
    
    def process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa colunas de data/hora"""
        date_columns = [
            "Data Coleta",
            "Data Hora Inicio Operacao",
            "Data Hora Final Operacao",
            "Data Hora Saida Terminal"
        ]
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Adiciona colunas derivadas de data
        if "Data Coleta" in df.columns:
            df["Data"] = df["Data Coleta"].dt.date
            df["Ano"] = df["Data Coleta"].dt.year
            df["Mes"] = df["Data Coleta"].dt.month
            df["Dia"] = df["Data Coleta"].dt.day
            df["DiaSemana"] = df["Data Coleta"].dt.day_name(locale='pt_BR')
            df["Hora"] = df["Data Coleta"].dt.hour
        
        return df
    
    def process_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa colunas numéricas"""
        numeric_columns = [
            "Passageiros", "Distancia", "Num Terminal Viagem",
            "Catraca Inicial", "Catraca Final", "Total Fichas",
            "Quant Gratuidade", "Quant Passagem", "Quant Passe",
            "Quant Vale Transporte", "Quant Inteiras"
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                # Converte formato brasileiro para float
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def process_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa colunas categóricas"""
        categorical_columns = [
            "Nome Linha", "Codigo Externo Linha", "Numero Veiculo",
            "Nome Garagem", "Descricao Terminal", "Periodo Operacao",
            "Tipo Viagem", "Cobrador/Operador", "Nome Operadora"
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
    
    def add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas derivadas úteis para análise"""
        
        # Hora base para heatmaps
        if "Data Hora Inicio Operacao" in df.columns:
            df["Hora_Base"] = pd.to_datetime(
                df["Data Hora Inicio Operacao"], 
                errors='coerce'
            ).dt.hour
            df["DiaSemana_Base"] = pd.to_datetime(
                df["Data Hora Inicio Operacao"],
                errors='coerce'
            ).dt.dayofweek
        
        # Duração da operação
        if all(col in df.columns for col in ["Data Hora Inicio Operacao", "Data Hora Final Operacao"]):
            inicio = pd.to_datetime(df["Data Hora Inicio Operacao"], errors='coerce')
            fim = pd.to_datetime(df["Data Hora Final Operacao"], errors='coerce')
            df["Duracao_Min"] = (fim - inicio).dt.total_seconds() / 60.0
            df["Duracao_Horas"] = df["Duracao_Min"] / 60.0
        
        # Total de pagantes
        cols_pagantes = [col for col in self.config.COLS_PAGANTES if col in df.columns]
        if cols_pagantes:
            df["Total_Pagantes"] = df[cols_pagantes].sum(axis=1)
        
        # Proporção gratuidade/pagantes
        if "Quant Gratuidade" in df.columns and "Total_Pagantes" in df.columns:
            df["Prop_Gratuidade"] = df["Quant Gratuidade"] / df["Total_Pagantes"].replace(0, np.nan)
        
        return df
    
    def find_column(self, 
                   df: pd.DataFrame,
                   candidates: List[str]) -> Optional[str]:
        """
        Encontra a primeira coluna existente de uma lista de candidatas
        
        Args:
            df: DataFrame
            candidates: Lista de nomes de colunas candidatas
            
        Returns:
            Nome da coluna encontrada ou None
        """
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def aggregate_by_line(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega dados por linha"""
        line_col = self.find_column(df, self.config.COLS_LINHA)
        
        if not line_col:
            return pd.DataFrame()
        
        agg_dict = {
            'Passageiros': 'sum',
            'Distancia': 'sum',
            'Total_Pagantes': 'sum',
            'Quant Gratuidade': 'sum'
        }
        
        # Remove colunas que não existem
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        return df.groupby(line_col).agg(agg_dict).reset_index()
    
    def aggregate_by_driver(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrega dados por motorista"""
        driver_col = self.find_column(df, self.config.COLS_MOTORISTA)
        
        if not driver_col:
            return pd.DataFrame()
        
        agg_dict = {
            'Passageiros': 'sum',
            'Distancia': 'sum',
            'Total_Pagantes': 'sum',
            'Duracao_Horas': 'sum'
        }
        
        # Remove colunas que não existem
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        result = df.groupby(driver_col).agg(agg_dict)
        result['Viagens'] = df.groupby(driver_col).size()
        
        return result.reset_index()