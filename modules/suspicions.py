# -*- coding: utf-8 -*-
"""Módulo de análises de suspeições (gratuidades, outliers, etc.)"""

from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Optional, Dict, Any

from config.settings import Config
from utils.formatters import BrazilianFormatter


class SuspicionAnalysis:
    """Análises de suspeições a partir de dados operacionais."""

    def __init__(self, df: pd.DataFrame, config: Optional[Config] = None):
        self.df = df
        self.config = config or Config()
        self.formatter = BrazilianFormatter()

    # ---------------------------- UI ENTRYPOINT ---------------------------- #
    def render(self) -> None:
        """Renderiza a aba de análise de suspeições."""
        st.header("🔎 Análise de Suspeições")

        driver_col = self._find_driver_column()
        if not driver_col:
            st.warning("Coluna de motorista não encontrada nos dados.")
            return

        with st.expander("Parâmetros da Análise", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                min_trips = st.number_input("Mínimo de viagens", min_value=1, max_value=1000, value=10, step=1)
            with col2:
                min_passengers = st.number_input("Mínimo de passageiros", min_value=0, max_value=1_000_000, value=100, step=10)
            with col3:
                z_threshold = st.number_input("Z-score de alerta (proporção de gratuidade)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

        self.analyze_gratuities(min_trips=min_trips, min_passengers=min_passengers, z_threshold=z_threshold)

    # ---------------------------- CORE ANALYSES ---------------------------- #
    def analyze_gratuities(self, min_trips: int, min_passengers: int, z_threshold: float) -> None:
        """Analisa taxas de gratuidade por motorista e destaca potenciais anomalias."""
        driver_col = self._find_driver_column()
        if not driver_col:
            st.info("Coluna de motorista não identificada para análise de gratuidades.")
            return

        ratios = self.calculate_gratuity_ratios(driver_col)
        if ratios.empty:
            st.info("Não há dados suficientes para calcular proporções de gratuidade.")
            return

        # Filtros mínimos
        ratios_f = ratios.copy()
        ratios_f = ratios_f[ratios_f["Viagens"] >= min_trips]
        if "Passageiros" in ratios_f.columns:
            ratios_f = ratios_f[ratios_f["Passageiros"].fillna(0) >= min_passengers]

        # Z-score da proporção de gratuidade
        prop = ratios_f["Prop_Gratuidade"].fillna(0)
        if len(prop) >= 2 and prop.std(ddof=0) > 0:
            zscores = (prop - prop.mean()) / prop.std(ddof=0)
        else:
            zscores = pd.Series([0] * len(prop), index=ratios_f.index)

        ratios_f["Z_Prop_Gratuidade"] = zscores
        suspeitos = ratios_f[ratios_f["Z_Prop_Gratuidade"] >= z_threshold].copy()

        # Ordenação e exibição
        if not suspeitos.empty:
            st.subheader("⚠️ Possíveis Anomalias (Alta proporção de gratuidades)")
            ordem = "Z_Prop_Gratuidade"
            suspeitos = suspeitos.sort_values(ordem, ascending=False)
            # Formatação amigável
            for col in ["Viagens", "Pagantes", "Gratuidades", "Passageiros"]:
                if col in suspeitos.columns:
                    suspeitos[col] = suspeitos[col].apply(self.formatter.format_integer)
            if "Prop_Gratuidade" in suspeitos.columns:
                suspeitos["Prop_Gratuidade"] = suspeitos["Prop_Gratuidade"].apply(lambda x: f"{x:.2%}")
            if "Z_Prop_Gratuidade" in suspeitos.columns:
                suspeitos["Z_Prop_Gratuidade"] = suspeitos["Z_Prop_Gratuidade"].apply(lambda x: f"{x:.2f}")
            st.dataframe(suspeitos.reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.success("Nenhuma anomalia relevante encontrada com os parâmetros atuais.")

        # Tabela resumida completa
        st.subheader("📋 Resumo por Motorista")
        vis = ratios.copy()
        for col in ["Viagens", "Pagantes", "Gratuidades", "Passageiros"]:
            if col in vis.columns:
                vis[col] = vis[col].apply(self.formatter.format_integer)
        if "Prop_Gratuidade" in vis.columns:
            vis["Prop_Gratuidade"] = vis["Prop_Gratuidade"].apply(lambda x: f"{x:.2%}")
        st.dataframe(vis, use_container_width=True, hide_index=True)

    # ---------------------------- COMPUTATIONS ---------------------------- #
    def calculate_gratuity_ratios(self, driver_col: str) -> pd.DataFrame:
        """Calcula proporções de gratuidade por motorista (robusto, sem colisão no reset_index)."""
        # Colunas de pagantes vindas da configuração
        paying_cols = [col for col in getattr(self.config, "COLS_PAGANTES", []) if col in self.df.columns]

        # Se não há pagantes nem coluna de gratuidade, aborta
        has_grat = ("Gratuidade" in self.df.columns) or ("Quant Gratuidade" in self.df.columns)
        if not paying_cols and not has_grat:
            return pd.DataFrame(columns=["Motorista", "Viagens", "Pagantes", "Gratuidades", "Prop_Gratuidade"])

        df = self.df.copy()

        # Harmoniza a coluna de gratuidade
        if "Gratuidade" not in df.columns and "Quant Gratuidade" in df.columns:
            df = df.rename(columns={"Quant Gratuidade": "Gratuidade"})

        # Coerção numérica
        numeric_cols = ["Gratuidade", "Passageiros", "Distancia"] + paying_cols
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Agrupamento
        gb = df.groupby(driver_col, dropna=False)

        # Viagens por grupo via size() evita duplicar a chave ao dar reset_index
        base = gb.size().rename("Viagens").reset_index()

        # Somatórios separados e merges
        result = base
        if "Gratuidade" in df.columns:
            grat_sum = gb["Gratuidade"].sum().reset_index(name="Gratuidades")
            result = result.merge(grat_sum, on=driver_col, how="left")
        if "Passageiros" in df.columns:
            pax_sum = gb["Passageiros"].sum().reset_index(name="Passageiros")
            result = result.merge(pax_sum, on=driver_col, how="left")
        if paying_cols:
            pag_sum = gb[paying_cols].sum().reset_index()
            result = result.merge(pag_sum, on=driver_col, how="left")

        # Total de pagantes e proporção
        if paying_cols:
            result["Pagantes"] = result[paying_cols].sum(axis=1, numeric_only=True)
        else:
            result["Pagantes"] = 0

        if "Gratuidades" not in result.columns:
            result["Gratuidades"] = 0

        denom = (result["Pagantes"] + result["Gratuidades"]).replace(0, np.nan)
        result["Prop_Gratuidade"] = (result["Gratuidades"] / denom).fillna(0)

        # Renomeia a coluna do motorista no final, evitando colisões
        result = result.rename(columns={driver_col: "Motorista"})

        # Ordena por maior proporção de gratuidade
        result = result.sort_values("Prop_Gratuidade", ascending=False)

        # Seleção de colunas apresentáveis
        cols = ["Motorista", "Viagens", "Pagantes", "Gratuidades", "Prop_Gratuidade"]
        extra = [c for c in ["Passageiros"] if c in result.columns]
        cols = cols[:2] + extra + cols[2:]
        result = result[cols]

        return result

    # ---------------------------- HELPERS ---------------------------- #
    def _find_driver_column(self) -> Optional[str]:
        """Identifica a coluna de motorista conforme a configuração e aliases comuns."""
        # 1) Config árvore
        for col in getattr(self.config, "COLS_MOTORISTA", []):
            if col in self.df.columns:
                return col

        # 2) Aliases frequentes
        aliases = [
            "Cobrador/Operador", "Motorista", "Nome Motorista", "Condutor",
            "Colaborador", "Funcionario", "Funcionário", "Operador", "Cobrador"
        ]
        for c in aliases:
            if c in self.df.columns:
                return c

        # 3) Heurística básica
        for c in self.df.columns:
            if "motor" in str(c).lower() or "operador" in str(c).lower():
                return c

        return None
