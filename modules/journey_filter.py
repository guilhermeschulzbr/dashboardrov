# -*- coding: utf-8 -*-
"""Filtro de jornadas por duração diária (threshold padrão: 13h)."""

import pandas as pd
from typing import Optional, List

DEFAULT_THRESHOLD_H = 13.0  # 13:00 horas

def _find_driver_column(df: pd.DataFrame, candidates_extra: Optional[List[str]] = None) -> Optional[str]:
    """Detecta a coluna do motorista por configuração/heurística leve."""
    candidates = (candidates_extra or []) + [
        "Cobrador/Operador", "Motorista", "Nome Motorista", "Condutor",
        "Colaborador", "Funcionario", "Funcionário", "Operador", "Cobrador"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        s = str(c).lower()
        if "motor" in s or "operador" in s:
            return c
    return None

def filter_by_daily_journey(
    df: pd.DataFrame,
    mode: str,
    threshold_hours: float = DEFAULT_THRESHOLD_H,
    driver_col: Optional[str] = None
) -> pd.DataFrame:
    """Filtra o DataFrame pelo total diário de horas por motorista.
    
    mode:
      - "Não filtrar": retorna df original
      - "Mostrar apenas": mantém somente jornadas > threshold_hours
      - "Expurgar": remove jornadas > threshold_hours
    """
    if df is None or df.empty:
        return df
    if mode not in {"Não filtrar", "Mostrar apenas", "Expurgar"}:
        return df
    if mode == "Não filtrar":
        return df.copy()

    drv_col = driver_col or _find_driver_column(df)
    if not drv_col or drv_col not in df.columns:
        return df

    tmp = df.copy()

    # Converte datas/horas de operação
    inicio_col = "Data Hora Inicio Operacao"
    fim_col = "Data Hora Final Operacao"
    if inicio_col not in tmp.columns or fim_col not in tmp.columns:
        return df

    tmp["_inicio"] = pd.to_datetime(tmp[inicio_col], errors="coerce")
    tmp["_fim"] = pd.to_datetime(tmp[fim_col], errors="coerce")
    tmp = tmp.dropna(subset=["_inicio", "_fim"])
    tmp = tmp[tmp["_fim"] > tmp["_inicio"]]

    tmp["_dia"] = tmp["_inicio"].dt.date
    tmp["_dur_h"] = (tmp["_fim"] - tmp["_inicio"]).dt.total_seconds() / 3600.0

    sums = (
        tmp.groupby([drv_col, "_dia"])["_dur_h"]
        .sum()
        .reset_index(name="_total_h")
    )
    tmp = tmp.merge(sums, on=[drv_col, "_dia"], how="left")

    if mode == "Mostrar apenas":
        tmp = tmp[tmp["_total_h"] > threshold_hours]
    elif mode == "Expurgar":
        tmp = tmp[tmp["_total_h"] <= threshold_hours]

    # Retorna somente as colunas originais
    return tmp[df.columns]
