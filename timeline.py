# -*- coding: utf-8 -*-
"""Módulo de análise de linha do tempo"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, datetime, timedelta
from typing import Optional
from config.settings import Config
from utils.formatters import BrazilianFormatter


def hours_to_hhmm(hours: float) -> str:
    """Converte horas decimais para formato HH:MM"""
    try:
        total = int(round(float(hours) * 60))
    except Exception:
        return "00:00"
    sign = "-" if total < 0 else ""
    total = abs(total)
    h, m = divmod(total, 60)
    return f"{sign}{h:02d}:{m:02d}"


def hhmm_to_hours(hhmm: str) -> float:
    """Converte formato HH:MM para horas decimais"""
    try:
        s = str(hhmm).strip()
        neg = s.startswith("-")
        if neg:
            s = s[1:]
        parts = s.split(":")
        if len(parts) != 2:
            return 0.0
        h, m = parts
        val = int(h) + int(m)/60.0
        return -val if neg else val
    except Exception:
        return 0.0


def date_input_br(label, value, key=None, min_value=None, max_value=None):
    """Wrapper para manter formato brasileiro sempre"""
    return st.date_input(
        label,
        value=value,
        min_value=min_value,
        max_value=max_value,
        format="DD/MM/YYYY",
        key=key
    )


def get_cell_color_and_icon(hours: float, is_negative: bool = False, total_hours_worked: float = 0) -> tuple:
    """
    Retorna cor e ícone baseado nas horas extras ou negativas
    
    Args:
        hours: Horas em decimal (extras ou negativas)
        is_negative: Se são horas negativas
        total_hours_worked: Total de horas trabalhadas no dia
        
    Returns:
        Tupla (cor_hex, ícone)
    """
    if is_negative:
        # Horas negativas - só considera se trabalhou pelo menos 1 hora
        if total_hours_worked < 1.0:
            return "#F0F0F0", ""  # Cinza - não se aplica
        
        if hours >= 2:  # Faltou 2h ou mais
            return "#FF6B6B", "🔴"  # Vermelho
        elif hours >= 1:  # Faltou entre 1h e 2h
            return "#FFA500", "🟠"  # Laranja
        elif hours > 0:  # Faltou menos de 1h
            return "#FFD700", "🟡"  # Amarelo
        else:
            return "#90EE90", "✅"  # Verde (cumpriu jornada)
    else:
        # Horas extras
        if hours >= 2:  # 2h ou mais de extra
            return "#FF4444", "⚠️"  # Vermelho forte
        elif hours >= 1:  # Entre 1h e 2h de extra
            return "#FF8C00", "⚡"  # Laranja
        elif hours > 0.0167:  # Mais de 1 minuto de extra
            return "#FFD700", "📍"  # Amarelo
        else:
            return "#F0F0F0", ""  # Cinza claro (sem extra)


def create_overtime_table_safe(df: pd.DataFrame, view_type: str = "extras", daily_hours: pd.DataFrame = None) -> None:
    """
    Cria tabela de horas extras/negativas usando Pandas Styler (mais seguro)
    
    Args:
        df: DataFrame com os dados formatados
        view_type: Tipo de visualização
        daily_hours: DataFrame com total de horas trabalhadas por motorista/dia
    """
    import pandas as pd
    
    # Prepara o DataFrame
    display_df = df.copy()
    
    # Cria mapa de horas trabalhadas se disponível
    hours_map = {}
    if daily_hours is not None:
        for _, row in daily_hours.iterrows():
            key = (row['Motorista'], row['Dia'])
            hours_map[key] = row['Horas (h)']
    
    # Função para aplicar cores
    def color_cell(val):
        if pd.isna(val) or val == '' or val == '-' or val == '00:00':
            return 'background-color: #F8F9FA; color: #999;'
        
        try:
            val_str = str(val)
            is_negative = '-' in val_str
            
            # Remove sinal e extrai horas
            val_abs = val_str.replace('-', '')
            if ':' in val_abs:
                h, m = val_abs.split(':')
                total_hours = int(h) + int(m)/60.0
                
                if is_negative:
                    # Horas negativas
                    if total_hours >= 2:
                        return 'background-color: #FF6B6B; color: white; font-weight: bold;'
                    elif total_hours >= 1:
                        return 'background-color: #FFA500; color: white; font-weight: bold;'
                    elif total_hours > 0:
                        return 'background-color: #FFD700; color: black; font-weight: bold;'
                else:
                    # Horas extras
                    if total_hours >= 2:
                        return 'background-color: #FF4444; color: white; font-weight: bold;'
                    elif total_hours >= 1:
                        return 'background-color: #FF8C00; color: white; font-weight: bold;'
                    elif total_hours > 0:
                        return 'background-color: #FFD700; color: black; font-weight: bold;'
        except:
            pass
        
        return ''
    
    # Aplica estilos
    styled = display_df.style
    
    # Aplica cores nas colunas de dados (exceto Motorista e TOTAL)
    for col in display_df.columns:
        if col not in ['Motorista', 'TOTAL']:
            styled = styled.applymap(color_cell, subset=[col])
    
    # Destaca coluna TOTAL
    if 'TOTAL' in display_df.columns:
        styled = styled.set_properties(
            subset=['TOTAL'],
            **{'font-weight': 'bold', 'background-color': '#D5DBDB'}
        )
    
    # Destaca coluna Motorista
    if 'Motorista' in display_df.columns:
        styled = styled.set_properties(
            subset=['Motorista'],
            **{'font-weight': 'bold', 'text-align': 'left', 'background-color': '#ECF0F1'}
        )
    
    # Exibe a tabela
    st.dataframe(styled, use_container_width=True, height=600)
    
    # Legenda
    st.markdown("""
    **Legenda:**
    
    **Horas Extras (valores positivos):**
    - 🔴 Vermelho: ≥ 2 horas extras
    - 🟠 Laranja: 1h - 1h59min extras
    - 🟡 Amarelo: < 1 hora extra
    
    **Horas Negativas (valores negativos, apenas para quem trabalhou ≥ 1h e < 7h20):**
    - 🔴 Vermelho: Faltou ≥ 2 horas para completar 7h20
    - 🟠 Laranja: Faltou 1h - 1h59min para completar 7h20
    - 🟡 Amarelo: Faltou < 1 hora para completar 7h20
    - ⬜ Cinza: Não se aplica (trabalhou < 1 hora no dia)
    """)


class TimelineAnalysis:
    """Análise de linha do tempo"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.config = Config()
        self.formatter = BrazilianFormatter()
        self.jornada_decimal = (7 * 60 + 20) / 60.0  # 7h20 em decimal
        self.minimo_para_negativa = 1.0  # Mínimo de 1 hora para considerar hora negativa
    
    def render(self):
        """Renderiza análise de linha do tempo"""
        st.header("📅 Linha do Tempo")
        
        # Verifica colunas necessárias
        required_cols = ['Data Hora Inicio Operacao', 'Data Hora Final Operacao']
        missing = [col for col in required_cols if col not in self.df.columns]
        
        if missing:
            st.error(f"Colunas necessárias não encontradas: {missing}")
            return
        
        # Tabs de visualização
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚌 Veículos x Linhas",
            "👤 Motoristas x Linhas",
            "📊 Ocupação da Frota",
            "📅 Horas Extras/Negativas (Período)"
        ])
        
        with tab1:
            self.render_vehicle_timeline()
        
        with tab2:
            self.render_driver_timeline()
        
        with tab3:
            self.render_fleet_occupation()
        
        with tab4:
            self.render_overtime_period_analysis()
    
    def render_vehicle_timeline(self):
        """Renderiza timeline de veículos"""
        st.subheader("🚌 Alocação de Veículos por Linha")
        
        # Controles de visualização
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Seletor de data
            selected_date = self.select_date()
        
        with col2:
            # Seletor de ordenação
            order_option = st.selectbox(
                "Ordenar por:",
                ["Cronológica (primeira viagem)", "Alfabética/Numérica", "Quantidade de viagens"],
                key="vehicle_order"
            )
        
        if not selected_date:
            return
        
        # Filtra dados do dia
        day_data = self.filter_by_date(selected_date)
        
        if day_data.empty:
            st.info(f"Sem dados para {selected_date}")
            return
        
        # Prepara dados para timeline
        timeline_data = self.prepare_vehicle_timeline_data(day_data)
        
        if timeline_data.empty:
            st.info("Sem dados válidos para criar a timeline")
            return
        
        # Converte duração para formato HH:MM
        timeline_data['Duração'] = timeline_data['Duração (min)'].apply(lambda x: hours_to_hhmm(x/60))
        
        # Garantir que Veículo seja tratado como string/categoria
        timeline_data['Veículo'] = timeline_data['Veículo'].astype(str)
        
        # APLICAR ORDENAÇÃO ESCOLHIDA
        if order_option == "Alfabética/Numérica":
            # Ordena alfabeticamente/numericamente
            vehicle_order = sorted(timeline_data['Veículo'].unique())
        elif order_option == "Quantidade de viagens":
            # Ordena por quantidade de viagens (decrescente)
            vehicle_trips = timeline_data.groupby('Veículo').size().sort_values(ascending=False)
            vehicle_order = vehicle_trips.index.tolist()
        else:  # Cronológica (padrão)
            # Ordena pela primeira aparição (hora de início)
            timeline_data = timeline_data.sort_values('Início')
            vehicle_first_appearance = timeline_data.groupby('Veículo')['Início'].min().sort_values()
            vehicle_order = vehicle_first_appearance.index.tolist()
        
        # Cria visualização
        fig = px.timeline(
            timeline_data,
            x_start="Início",
            x_end="Fim",
            y="Veículo",
            color="Linha",
            title=f"",
            hover_data=["Duração", "Passageiros"],
            category_orders={"Veículo": vehicle_order}  # Define ordem específica
        )
        
        # CONFIGURAÇÃO CORRETA: Forçar eixo Y como categoria sempre
        fig.update_yaxes(
            type='category',  # SEMPRE categoria, nunca numérico
            categoryorder='array',  # Usar ordem do array
            categoryarray=vehicle_order,  # Array com ordem desejada
            autorange="reversed"  # Inverter para mostrar primeiro no topo
        )
        
        # Adiciona linha vertical para hora atual se for hoje
        if selected_date == date.today():
            now = datetime.now()
            fig.add_vline(
                x=now,
                line_dash="dash",
                line_color="red",
                annotation_text="Agora"
            )
        
        # Ajusta layout
        fig.update_layout(
            height=max(600, len(vehicle_order) * 30),  # Altura dinâmica
            xaxis_title="Horário",
            yaxis_title="Veículo",
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Formata eixo X para mostrar horas
        fig.update_xaxes(
            tickformat="%H:%M",
            dtick=3600000,  # Tick a cada hora
            rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Informações adicionais
        with st.expander("ℹ️ Informações da Timeline"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Ordenação Atual:**")
                if order_option == "Alfabética/Numérica":
                    st.markdown("Veículos em ordem alfabética/numérica")
                elif order_option == "Quantidade de viagens":
                    st.markdown("Mais viagens primeiro")
                else:
                    st.markdown("Ordem de início da primeira viagem")
            
            with col2:
                st.markdown("**Cores:**")
                st.markdown("Cada cor representa uma linha diferente")
            
            with col3:
                st.markdown("**Interação:**")
                st.markdown("Passe o mouse sobre as barras para ver detalhes")
        
        # Métricas do dia
        self.render_day_metrics(day_data)
    
    def render_driver_timeline(self):
        """Renderiza timeline de motoristas"""
        st.subheader("👤 Jornada de Motoristas")
        
        driver_col = self._find_driver_column()
        
        if not driver_col:
            st.warning("Coluna de motorista não encontrada")
            return
        
        # Controles de visualização
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Seletor de data
            selected_date = self.select_date(key="driver_date")
        
        with col2:
            # Seletor de ordenação
            order_option = st.selectbox(
                "Ordenar por:",
                ["Cronológica (primeira viagem)", "Alfabética", "Volume de horas"],
                key="driver_order"
            )
        
        if not selected_date:
            return
        
        # Filtra dados do dia
        day_data = self.filter_by_date(selected_date)
        
        if day_data.empty:
            st.info(f"Sem dados para {selected_date}")
            return
        
        # Prepara dados para timeline
        timeline_data = self.prepare_driver_timeline_data(day_data, driver_col)
        
        if timeline_data.empty:
            st.info("Sem dados válidos para criar a timeline")
            return
        
        # Converte duração para string HH:MM e calcula HE/negativa por motorista
        timeline_data['Duração'] = timeline_data['Duração (h)'].apply(hours_to_hhmm)
        
        # Calcula totais por motorista
        _daily = (timeline_data.groupby('Motorista', as_index=False)['Duração (h)']
                 .sum()
                 .rename(columns={'Duração (h)': 'HorasDia'}))
        _daily['Extra (h)'] = (_daily['HorasDia'] - self.jornada_decimal).clip(lower=0)
        _daily['Negativa (h)'] = _daily.apply(
            lambda row: max(0, self.jornada_decimal - row['HorasDia'])
                        if row['HorasDia'] >= self.minimo_para_negativa
                        else 0,
            axis=1
        )
        
        # Mapeia valores para cada linha
        _map_h = dict(zip(_daily['Motorista'], _daily['HorasDia']))
        _map_e = dict(zip(_daily['Motorista'], _daily['Extra (h)']))
        _map_n = dict(zip(_daily['Motorista'], _daily['Negativa (h)']))
        
        timeline_data['Extra (h)'] = timeline_data['Motorista'].map(_map_e).fillna(0.0)
        timeline_data['Negativa (h)'] = timeline_data['Motorista'].map(_map_n).fillna(0.0)
        timeline_data['Extra'] = timeline_data['Extra (h)'].apply(hours_to_hhmm)
        timeline_data['Negativa'] = timeline_data['Negativa (h)'].apply(hours_to_hhmm)
        
        # Rótulo exibido no eixo Y com indicadores
        def _mk_label(name):
            h = _map_h.get(name, 0.0)
            e = _map_e.get(name, 0.0)
            n = _map_n.get(name, 0.0)
            
            # Define ícone baseado na situação
            if e >= 2:
                badge = ' ⚠️'
            elif e >= 1:
                badge = ' ⚡'
            elif e > 0:
                badge = ' 📍'
            elif n > 0 and h >= self.minimo_para_negativa:
                if n >= 2:
                    badge = ' 🔴'
                elif n >= 1:
                    badge = ' 🟠'
                else:
                    badge = ' 🟡'
            elif h < self.minimo_para_negativa:
                badge = ' ⬜'  # Menos de 1h trabalhada
            else:
                badge = ' ✅'
            
            core = f"{name} • {hours_to_hhmm(h)}"
            
            if e > 0:
                tail = f" • HE {hours_to_hhmm(e)}"
            elif n > 0 and h >= self.minimo_para_negativa:
                tail = f" • NEG {hours_to_hhmm(n)}"
            elif h < self.minimo_para_negativa:
                tail = " • < 1h"
            else:
                tail = ""
            
            return core + tail + badge
        
        timeline_data['MotoristaExib'] = timeline_data['Motorista'].astype(str).map(_mk_label)
        
        # APLICAR ORDENAÇÃO ESCOLHIDA
        if order_option == "Alfabética":
            # Ordena alfabeticamente
            motorista_order = sorted(timeline_data['MotoristaExib'].unique())
        elif order_option == "Volume de horas":
            # Ordena por total de horas trabalhadas (decrescente)
            daily_sorted = _daily.sort_values('HorasDia', ascending=False)
            motorista_order = [_mk_label(m) for m in daily_sorted['Motorista']]
        else:  # Cronológica (padrão)
            # Ordena pela primeira aparição (hora de início)
            first_appearance = timeline_data.groupby('MotoristaExib')['Início'].min().sort_values()
            motorista_order = first_appearance.index.tolist()
        
        # Cria visualização
        fig = px.timeline(
            timeline_data,
            x_start="Início",
            x_end="Fim",
            y="MotoristaExib",
            color="Linha",
            title=f"Jornada de Motoristas - {selected_date} ({order_option})",
            hover_data=["Duração", "Extra", "Negativa", "Passageiros"],
            category_orders={"MotoristaExib": motorista_order}  # Aplica ordem escolhida
        )
        
        fig.update_yaxes(
            categoryorder='array',
            categoryarray=motorista_order,
            autorange="reversed"
        )
        fig.update_layout(height=max(600, len(motorista_order) * 25))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Informações sobre a ordenação
        with st.expander("ℹ️ Informações da Timeline"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Ordenação Atual:**")
                if order_option == "Alfabética":
                    st.markdown("Motoristas em ordem alfabética")
                elif order_option == "Volume de horas":
                    st.markdown("Maior jornada primeiro")
                else:
                    st.markdown("Ordem de início da primeira viagem")
            
            with col2:
                st.markdown("**Indicadores:**")
                st.markdown("⚠️ HE ≥ 2h | ⚡ HE 1-2h | 📍 HE < 1h")
                st.markdown("🔴 Neg ≥ 2h | 🟠 Neg 1-2h | 🟡 Neg < 1h")
            
            with col3:
                st.markdown("**Cores:**")
                st.markdown("Cada cor = uma linha diferente")
                st.markdown("⬜ = Trabalhou menos de 1h")
        
        # Análise de horas extras do dia
        self.render_overtime_analysis_day(_daily)
    
    def render_fleet_occupation(self):
        """Renderiza ocupação da frota ao longo do tempo"""
        st.subheader("📊 Ocupação da Frota")
        
        # Seletor de período
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = date_input_br(
                "Data Inicial",
                value=self.df['Data'].min() if 'Data' in self.df.columns else date.today(),
                key="fleet_start"
            )
        
        with col2:
            end_date = date_input_br(
                "Data Final",
                value=self.df['Data'].max() if 'Data' in self.df.columns else date.today(),
                key="fleet_end"
            )
        
        # Filtra período
        mask = (
            (pd.to_datetime(self.df['Data Hora Inicio Operacao']).dt.date >= start_date) &
            (pd.to_datetime(self.df['Data Hora Final Operacao']).dt.date <= end_date)
        )
        period_data = self.df[mask]
        
        if period_data.empty:
            st.info("Sem dados no período selecionado")
            return
        
        # Calcula ocupação por hora
        occupation_data = self.calculate_fleet_occupation(period_data)
        
        # Cria gráfico
        fig = px.area(
            occupation_data,
            x='Hora',
            y='Veículos em Operação',
            title='Ocupação Média da Frota por Hora',
            labels={'Hora': 'Hora do Dia', 'Veículos em Operação': 'Quantidade de Veículos'}
        )
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estatísticas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Pico de Ocupação",
                f"{occupation_data['Veículos em Operação'].max():.0f} veículos"
            )
        
        with col2:
            st.metric(
                "Horário de Pico",
                f"{occupation_data.loc[occupation_data['Veículos em Operação'].idxmax(), 'Hora']:.0f}h"
            )
        
        with col3:
            st.metric(
                "Ocupação Média",
                f"{occupation_data['Veículos em Operação'].mean():.1f} veículos"
            )
    
    def render_overtime_period_analysis(self):
        """Renderiza análise de horas extras e negativas por período com cores"""
        st.subheader("📅 Análise de Horas Extras e Negativas por Período")
        
        st.info("""
        ℹ️ **Regras de cálculo:**
        - **Horas Extras**: Tempo trabalhado acima de 7h20
        - **Horas Negativas**: Diferença para completar 7h20 (apenas para quem trabalhou ≥ 1h e < 7h20)
        - Motoristas que trabalharam menos de 1 hora no dia não têm horas negativas contabilizadas
        """)
        
        driver_col = self._find_driver_column()
        
        if not driver_col:
            st.warning("Coluna de motorista não encontrada")
            return
        
        # Verifica se há dados válidos
        ini_col = 'Data Hora Inicio Operacao'
        fim_col = 'Data Hora Final Operacao'
        
        if ini_col not in self.df.columns or fim_col not in self.df.columns:
            st.error("Colunas de horário não encontradas")
            return
        
        # Determina range de datas
        s_all = pd.to_datetime(self.df[ini_col], errors="coerce").dropna()
        if s_all.empty:
            st.info("Sem datas válidas no dataset")
            return
        
        min_date = s_all.min().date()
        max_date = s_all.max().date()
        
        # Seletores de período
        col1, col2 = st.columns(2)
        
        with col1:
            sel_start = date_input_br(
                "Data inicial",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="he_period_start"
            )
        
        with col2:
            sel_end = date_input_br(
                "Data final",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="he_period_end"
            )
        
        if sel_start > sel_end:
            st.warning("Período inválido: data inicial maior que a final")
            return
        
        # Filtra período
        mask = (
            (pd.to_datetime(self.df[ini_col]).dt.date >= sel_start) &
            (pd.to_datetime(self.df[ini_col]).dt.date <= sel_end)
        )
        period_df = self.df[mask].copy()
        
        if period_df.empty:
            st.info("Sem viagens no período selecionado")
            return
        
        # Calcula horas por motorista por dia
        daily_period = self._compute_overtime_daily_complete(period_df, driver_col)
        
        if daily_period.empty:
            st.info("Sem dados para análise no período")
            return
        
        # KPIs do período
        st.markdown("### 📊 Resumo do Período")
        
        # Filtra apenas motoristas que trabalharam >= 1h para contabilizar negativas
        motoristas_validos_neg = daily_period[
            (daily_period['Horas (h)'] >= self.minimo_para_negativa) &
            (daily_period['Negativa (h)'] > 0)
        ]
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        total_he = daily_period['Extra (h)'].sum()
        total_neg = motoristas_validos_neg['Negativa (h)'].sum()
        motoristas_com_he = daily_period[daily_period['Extra (h)'] > 0]['Motorista'].nunique()
        motoristas_com_neg = motoristas_validos_neg['Motorista'].nunique()
        media_he = daily_period[daily_period['Extra (h)'] > 0]['Extra (h)'].mean() if motoristas_com_he > 0 else 0
        media_neg = motoristas_validos_neg['Negativa (h)'].mean() if len(motoristas_validos_neg) > 0 else 0
        
        with col1:
            st.metric("Total HE", hours_to_hhmm(total_he))
        
        with col2:
            st.metric("Motoristas c/ HE", f"{motoristas_com_he}")
        
        with col3:
            st.metric("Média HE", hours_to_hhmm(media_he))
        
        with col4:
            st.metric("Total Negativas", hours_to_hhmm(total_neg))
        
        with col5:
            st.metric("Motoristas c/ Neg", f"{motoristas_com_neg}")
        
        with col6:
            st.metric("Média Neg", hours_to_hhmm(media_neg))
        
        # Opção de visualização
        st.markdown("### 📋 Tabela de Horas por Dia")
        
        view_option = st.radio(
            "Selecione o tipo de visualização:",
            ["Horas Extras", "Horas Negativas", "Saldo (Extras - Negativas)"],
            horizontal=True
        )
        
        # Prepara pivot table baseada na opção
        if view_option == "Horas Extras":
            pivot_df = daily_period.pivot_table(
                index='Motorista',
                columns='Dia',
                values='Extra (h)',
                aggfunc='sum',
                fill_value=0.0
            )
            table_title = "Horas Extras por Motorista e Dia"
        elif view_option == "Horas Negativas":
            pivot_df = daily_period.pivot_table(
                index='Motorista',
                columns='Dia',
                values='Negativa (h)',
                aggfunc='sum',
                fill_value=0.0
            )
            # Converte para negativo para exibição
            pivot_df = -pivot_df
            table_title = "Horas Negativas por Motorista e Dia (apenas ≥ 1h trabalhada)"
        else:  # Saldo
            pivot_extras = daily_period.pivot_table(
                index='Motorista',
                columns='Dia',
                values='Extra (h)',
                aggfunc='sum',
                fill_value=0.0
            )
            pivot_neg = daily_period.pivot_table(
                index='Motorista',
                columns='Dia',
                values='Negativa (h)',
                aggfunc='sum',
                fill_value=0.0
            )
            pivot_df = pivot_extras - pivot_neg
            table_title = "Saldo de Horas (Extras - Negativas)"
        
        # Adiciona coluna de total
        pivot_df['TOTAL'] = pivot_df.sum(axis=1)
        
        # Ordena por total (absoluto para considerar negativos)
        pivot_df = pivot_df.reindex(pivot_df['TOTAL'].abs().sort_values(ascending=False).index)
        
        # Formata para HH:MM mantendo sinal
        pivot_formatted = pivot_df.copy()
        
        # Reseta índice para ter Motorista como coluna
        pivot_formatted = pivot_formatted.reset_index()
        
        # Formata valores
        for col in pivot_formatted.columns:
            if col != 'Motorista':
                pivot_formatted[col] = pivot_formatted[col].apply(hours_to_hhmm)
        
        # Exibe tabela com estilos
        st.markdown(f"**{table_title}**")
        
        # Usa a versão segura da tabela
        create_overtime_table_safe(pivot_formatted, view_type=view_option, daily_hours=daily_period)
        
        # Botão de download
        csv = pivot_formatted.to_csv(sep=';', decimal=',', index=False).encode('utf-8-sig')
        st.download_button(
            "📥 Baixar tabela (CSV)",
            data=csv,
            file_name=f"horas_{view_option.lower().replace(' ', '_')}_{sel_start}_{sel_end}.csv",
            mime="text/csv"
        )
        
        # Gráficos de evolução
        st.markdown("### 📈 Evolução no Período")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de horas extras
            daily_he = daily_period.groupby('Dia')['Extra (h)'].sum().reset_index()
            daily_he['Extra'] = daily_he['Extra (h)'].apply(hours_to_hhmm)
            
            fig_he = px.bar(
                daily_he,
                x='Dia',
                y='Extra (h)',
                title='Total de Horas Extras por Dia',
                text='Extra',
                color_discrete_sequence=['#FF8C00']
            )
            
            fig_he.update_traces(textposition='outside')
            fig_he.update_layout(
                xaxis_tickformat='%d/%m',
                yaxis_title='Horas Extras',
                height=350
            )
            
            st.plotly_chart(fig_he, use_container_width=True)
        
        with col2:
            # Gráfico de horas negativas (apenas válidas)
            daily_neg = motoristas_validos_neg.groupby('Dia')['Negativa (h)'].sum().reset_index()
            daily_neg['Negativa'] = daily_neg['Negativa (h)'].apply(hours_to_hhmm)
            
            fig_neg = px.bar(
                daily_neg,
                x='Dia',
                y='Negativa (h)',
                title='Total de Horas Negativas por Dia (apenas ≥ 1h)',
                text='Negativa',
                color_discrete_sequence=['#FF6B6B']
            )
            
            fig_neg.update_traces(textposition='outside')
            fig_neg.update_layout(
                xaxis_tickformat='%d/%m',
                yaxis_title='Horas Negativas',
                height=350
            )
            
            st.plotly_chart(fig_neg, use_container_width=True)
    
    def render_overtime_analysis_day(self, daily_data: pd.DataFrame):
        """Análise de horas extras e negativas para um dia específico"""
        st.subheader("⏰ Análise de Horas do Dia")
        
        if daily_data.empty:
            st.info("Sem dados para análise")
            return
        
        # KPIs do dia
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        motoristas_com_he = len(daily_data[daily_data['Extra (h)'] > 0])
        motoristas_com_neg = len(daily_data[
            (daily_data['Negativa (h)'] > 0) &
            (daily_data['HorasDia'] >= self.minimo_para_negativa)
        ])
        total_he = daily_data['Extra (h)'].sum()
        total_neg = daily_data[
            daily_data['HorasDia'] >= self.minimo_para_negativa
        ]['Negativa (h)'].sum()
        media_he = daily_data[daily_data['Extra (h)'] > 0]['Extra (h)'].mean() if motoristas_com_he > 0 else 0
        media_neg = daily_data[
            (daily_data['Negativa (h)'] > 0) &
            (daily_data['HorasDia'] >= self.minimo_para_negativa)
        ]['Negativa (h)'].mean() if motoristas_com_neg > 0 else 0
        
        with col1:
            st.metric("Motoristas c/ HE", f"{motoristas_com_he}")
        
        with col2:
            st.metric("Total HE", hours_to_hhmm(total_he))
        
        with col3:
            st.metric("Média HE", hours_to_hhmm(media_he))
        
        with col4:
            st.metric("Motoristas c/ Neg", f"{motoristas_com_neg}")
        
        with col5:
            st.metric("Total Neg", hours_to_hhmm(total_neg))
        
        with col6:
            st.metric("Média Neg", hours_to_hhmm(media_neg))
        
        # Tabelas lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            if motoristas_com_he > 0:
                st.markdown("**🔴 Top 10 - Maior Hora Extra**")
                
                top10_he = daily_data.nlargest(10, 'Extra (h)')[['Motorista', 'HorasDia', 'Extra (h)']].copy()
                top10_he['Horas Trabalhadas'] = top10_he['HorasDia'].apply(hours_to_hhmm)
                top10_he['Hora Extra'] = top10_he['Extra (h)'].apply(hours_to_hhmm)
                
                # Adiciona ícones baseados no tempo de extra
                def add_icon_he(hours):
                    if hours >= 2:
                        return "⚠️"
                    elif hours >= 1:
                        return "⚡"
                    else:
                        return "📍"
                
                top10_he['Status'] = top10_he['Extra (h)'].apply(add_icon_he)
                
                st.dataframe(
                    top10_he[['Status', 'Motorista', 'Horas Trabalhadas', 'Hora Extra']],
                    hide_index=True
                )
            else:
                st.info("Nenhum motorista com hora extra")
        
        with col2:
            if motoristas_com_neg > 0:
                st.markdown("**🟡 Top 10 - Maior Hora Negativa**")
                
                # Filtra apenas quem tem hora negativa válida
                neg_validas = daily_data[
                    (daily_data['Negativa (h)'] > 0) &
                    (daily_data['HorasDia'] >= self.minimo_para_negativa)
                ]
                
                if not neg_validas.empty:
                    top10_neg = neg_validas.nlargest(10, 'Negativa (h)')[['Motorista', 'HorasDia', 'Negativa (h)']].copy()
                    top10_neg['Horas Trabalhadas'] = top10_neg['HorasDia'].apply(hours_to_hhmm)
                    top10_neg['Hora Negativa'] = top10_neg['Negativa (h)'].apply(lambda x: hours_to_hhmm(-x))
                    
                    # Adiciona ícones baseados no tempo negativo
                    def add_icon_neg(hours):
                        if hours >= 2:
                            return "🔴"
                        elif hours >= 1:
                            return "🟠"
                        else:
                            return "🟡"
                    
                    top10_neg['Status'] = top10_neg['Negativa (h)'].apply(add_icon_neg)
                    
                    st.dataframe(
                        top10_neg[['Status', 'Motorista', 'Horas Trabalhadas', 'Hora Negativa']],
                        hide_index=True
                    )
                else:
                    st.info("Nenhum motorista com hora negativa válida")
            else:
                st.info("Todos os motoristas cumpriram a jornada mínima ou trabalharam < 1h")
    
    def _compute_overtime_daily_complete(self, df_period: pd.DataFrame, driver_col: str) -> pd.DataFrame:
        """
        Calcula horas, extras e negativas por motorista por dia
        Hora negativa só é considerada se trabalhou >= 1h e < 7h20
        """
        if df_period.empty or not driver_col:
            return pd.DataFrame(columns=["Motorista", "Dia", "Horas (h)", "Extra (h)", "Negativa (h)"])
        
        ini_col = 'Data Hora Inicio Operacao'
        fim_col = 'Data Hora Final Operacao'
        
        if ini_col not in df_period.columns or fim_col not in df_period.columns:
            return pd.DataFrame(columns=["Motorista", "Dia", "Horas (h)", "Extra (h)", "Negativa (h)"])
        
        # Copia e processa dados
        dfp = df_period.copy()
        dfp[ini_col] = pd.to_datetime(dfp[ini_col], errors="coerce")
        dfp[fim_col] = pd.to_datetime(dfp[fim_col], errors="coerce")
        
        # Remove registros inválidos
        dfp = dfp.dropna(subset=[ini_col, fim_col, driver_col])
        dfp = dfp[dfp[fim_col] > dfp[ini_col]]
        
        if dfp.empty:
            return pd.DataFrame(columns=["Motorista", "Dia", "Horas (h)", "Extra (h)", "Negativa (h)"])
        
        # Calcula duração e dia
        dfp['Dia'] = dfp[ini_col].dt.date
        dfp['dur_h'] = (dfp[fim_col] - dfp[ini_col]).dt.total_seconds() / 3600.0
        
        # Agrupa por motorista e dia
        agg = dfp.groupby([driver_col, 'Dia'])['dur_h'].sum().reset_index()
        agg.columns = ['Motorista', 'Dia', 'Horas (h)']
        
        # Calcula hora extra (acima de 7h20)
        agg['Extra (h)'] = (agg['Horas (h)'] - self.jornada_decimal).clip(lower=0)
        
        # Calcula hora negativa (abaixo de 7h20, mas só se trabalhou >= 1h)
        agg['Negativa (h)'] = agg.apply(
            lambda row: max(0, self.jornada_decimal - row['Horas (h)'])
                        if row['Horas (h)'] >= self.minimo_para_negativa
                        else 0,
            axis=1
        )
        
        return agg
    
    def select_date(self, key: str = "timeline_date") -> Optional[date]:
        """Widget para seleção de data"""
        if 'Data' in self.df.columns:
            min_date = self.df['Data'].min()
            max_date = self.df['Data'].max()
            
            return date_input_br(
                "Selecione a data",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key=key
            )
        return None
    
    def filter_by_date(self, selected_date: date) -> pd.DataFrame:
        """Filtra dados por data"""
        start_time = pd.Timestamp(selected_date)
        end_time = start_time + timedelta(days=1)
        
        mask = (
            (pd.to_datetime(self.df['Data Hora Inicio Operacao']) >= start_time) &
            (pd.to_datetime(self.df['Data Hora Inicio Operacao']) < end_time)
        )
        
        return self.df[mask].copy()
    
    def prepare_vehicle_timeline_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara dados para timeline de veículos"""
        timeline_data = []
        
        for _, row in df.iterrows():
            inicio = pd.to_datetime(row['Data Hora Inicio Operacao'])
            fim = pd.to_datetime(row['Data Hora Final Operacao'])
            
            if pd.isna(inicio) or pd.isna(fim) or fim <= inicio:
                continue
            
            duracao_min = (fim - inicio).total_seconds() / 60
            
            timeline_data.append({
                'Veículo': str(row.get('Numero Veiculo', 'N/A')),
                'Linha': str(row.get('Nome Linha', 'N/A')),
                'Início': inicio,
                'Fim': fim,
                'Duração (min)': round(duracao_min, 1),
                'Passageiros': row.get('Passageiros', 0)
            })
        
        return pd.DataFrame(timeline_data)
    
    def prepare_driver_timeline_data(self, df: pd.DataFrame, driver_col: str) -> pd.DataFrame:
        """Prepara dados para timeline de motoristas"""
        timeline_data = []
        
        for _, row in df.iterrows():
            driver = str(row.get(driver_col, 'N/A'))
            inicio = pd.to_datetime(row['Data Hora Inicio Operacao'])
            fim = pd.to_datetime(row['Data Hora Final Operacao'])
            
            if pd.isna(inicio) or pd.isna(fim) or fim <= inicio:
                continue
            
            duracao_h = (fim - inicio).total_seconds() / 3600
            
            timeline_data.append({
                'Motorista': driver,
                'Linha': str(row.get('Nome Linha', 'N/A')),
                'Início': inicio,
                'Fim': fim,
                'Duração (h)': round(duracao_h, 2),
                'Passageiros': row.get('Passageiros', 0)
            })
        
        return pd.DataFrame(timeline_data)
    
    def calculate_fleet_occupation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula ocupação da frota por hora"""
        occupation_by_hour = []
        
        for hour in range(24):
            vehicles_in_operation = 0
            
            for _, row in df.iterrows():
                inicio = pd.to_datetime(row['Data Hora Inicio Operacao'])
                fim = pd.to_datetime(row['Data Hora Final Operacao'])
                
                if pd.isna(inicio) or pd.isna(fim):
                    continue
                
                # Verifica se o veículo estava operando nesta hora
                if inicio.hour <= hour < fim.hour:
                    vehicles_in_operation += 1
            
            # Calcula média por dia
            unique_days = len(df['Data'].unique()) if 'Data' in df.columns else 1
            
            occupation_by_hour.append({
                'Hora': hour,
                'Veículos em Operação': vehicles_in_operation / max(unique_days, 1)
            })
        
        return pd.DataFrame(occupation_by_hour)
    
    def render_day_metrics(self, df: pd.DataFrame):
        """Renderiza métricas do dia"""
        st.subheader("📊 Métricas do Dia")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_trips = len(df)
            st.metric("Total de Viagens", self.formatter.format_integer(total_trips))
        
        with col2:
            if 'Numero Veiculo' in df.columns:
                unique_vehicles = df['Numero Veiculo'].nunique()
                st.metric("Veículos Ativos", self.formatter.format_integer(unique_vehicles))
        
        with col3:
            if 'Passageiros' in df.columns:
                total_passengers = df['Passageiros'].sum()
                st.metric("Total de Passageiros", self.formatter.format_integer(total_passengers))
        
        with col4:
            if 'Distancia' in df.columns:
                total_km = df['Distancia'].sum()
                st.metric("KM Rodados", self.formatter.format_number(total_km, 1))
    
    def _find_driver_column(self) -> Optional[str]:
        """Encontra a coluna de motorista"""
        for col in self.config.COLS_MOTORISTA:
            if col in self.df.columns:
                return col
        return None
