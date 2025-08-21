# -*- coding: utf-8 -*-
"""Componente de tabelas"""


import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from utils.formatters import BrazilianFormatter
from config.settings import Config


class TableComponent:
    """Componente para renderização de tabelas"""
    
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df
        self.config = config
        self.formatter = BrazilianFormatter()
    
    def render_consolidated_table(self):
        """Renderiza tabela consolidada por linha"""
        if "Nome Linha" not in self.df.columns:
            st.info("Coluna 'Nome Linha' não encontrada")
            return
        
        df = self.df.copy()
        # Converte colunas numéricas
        num_candidates = ["Passageiros", "Distancia", "Gratuidade", "Quant Gratuidade"] + list(self.config.COLS_PAGANTES)
        for c in num_candidates:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # Named aggregation
        named_aggs = {}
        if "Passageiros" in df.columns:
            named_aggs["Passageiros"] = ("Passageiros", "sum")
        if "Distancia" in df.columns:
            named_aggs["Distancia"] = ("Distancia", "sum")
        if "Gratuidade" in df.columns:
            named_aggs["Gratuidade"] = ("Gratuidade", "sum")
        elif "Quant Gratuidade" in df.columns:
            named_aggs["Gratuidade"] = ("Quant Gratuidade", "sum")
        
        cols_pagantes = [col for col in self.config.COLS_PAGANTES if col in df.columns]
        for col in cols_pagantes:
            named_aggs[col] = (col, "sum")
        
        gb = df.groupby("Nome Linha", dropna=False)
        if named_aggs:
            table = gb.agg(**named_aggs).reset_index()
        else:
            table = gb.size().rename("Viagens").reset_index()
        
        # Viagens via size
        viagens = gb.size().rename("Viagens").reset_index()
        if "Viagens" in table.columns:
            table = table.drop(columns=["Viagens"])
        table = table.merge(viagens, on="Nome Linha", how="left")
        
        # Derivadas
        if "Passageiros" in table.columns and "Viagens" in table.columns:
            table["Média Pax/Viagem"] = table["Passageiros"] / table["Viagens"].replace(0, np.nan)
        if "Passageiros" in table.columns and "Distancia" in table.columns:
            table["IPK"] = table["Passageiros"] / table["Distancia"].replace(0, np.nan)
        
        # Totais pagantes e receita
        if cols_pagantes:
            table["Total Pagantes"] = table[cols_pagantes].sum(axis=1, numeric_only=True)
            table["Receita Total"] = table["Total Pagantes"].fillna(0) * (self.config.TARIFA_USUARIO + self.config.SUBSIDIO_PAGANTE)
        
        sort_col = "Receita Total" if "Receita Total" in table.columns else ("Passageiros" if "Passageiros" in table.columns else "Viagens")
        table = table.sort_values(sort_col, ascending=False)
        
        # Render
        st.dataframe(table, use_container_width=True, hide_index=True)
        
        csv = table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Baixar tabela (CSV)",
            data=csv,
            file_name="tabela_consolidada_linhas.csv",
            mime="text/csv"
        )
