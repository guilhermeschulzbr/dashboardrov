# -*- coding: utf-8 -*-
"""Validadores de dados"""


import pandas as pd
from typing import List, Dict, Any, Tuple
from config.settings import Config


class DataValidator:
    """Validador de dados do sistema"""
    
    def __init__(self):
        self.config = Config()
        self.errors = []
        self.warnings = []
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Valida DataFrame completo
        
        Returns:
            Tupla (válido, dict com erros e warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Validações
        self.validate_required_columns(df)
        self.validate_data_types(df)
        self.validate_data_quality(df)
        self.validate_business_rules(df)
        
        return len(self.errors) == 0, {
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def validate_required_columns(self, df: pd.DataFrame):
        """Valida presença de colunas obrigatórias"""
        # Pelo menos uma coluna de cada tipo deve existir
        essential_groups = {
            'Motorista': self.config.COLS_MOTORISTA,
            'Veículo': self.config.COLS_VEICULO,
            'Linha': self.config.COLS_LINHA,
            'Data': self.config.COLS_DATA
        }
        
        for group_name, columns in essential_groups.items():
            if not any(col in df.columns for col in columns):
                self.warnings.append(
                    f"Nenhuma coluna de {group_name} encontrada. "
                    f"Esperadas: {', '.join(columns)}"
                )
    
    def validate_data_types(self, df: pd.DataFrame):
        """Valida tipos de dados"""
        # Valida datas
        date_columns = [
            col for col in df.columns
            if any(date_col in col for date_col in ['Data', 'Hora'])
        ]
        
        for col in date_columns:
            if col in df.columns:
                # Tenta converter para datetime
                try:
                    pd.to_datetime(df[col], errors='coerce')
                    null_count = df[col].isna().sum()
                    if null_count > len(df) * 0.5:
                        self.warnings.append(
                            f"Coluna '{col}' tem {null_count} valores inválidos "
                            f"({null_count/len(df)*100:.1f}%)"
                        )
                except Exception:
                    self.errors.append(f"Coluna '{col}' não pode ser convertida para data/hora")
        
        # Valida numéricos
        numeric_columns = ['Passageiros', 'Distancia', 'Quant Gratuidade'] + self.config.COLS_PAGANTES
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    null_count = df[col].isna().sum()
                    if null_count > len(df) * 0.3:
                        self.warnings.append(
                            f"Coluna numérica '{col}' tem muitos valores inválidos "
                            f"({null_count/len(df)*100:.1f}%)"
                        )
                except Exception:
                    self.errors.append(f"Coluna '{col}' deveria ser numérica")
    
    def validate_data_quality(self, df: pd.DataFrame):
        """Valida qualidade dos dados"""
        # Verifica se há dados
        if len(df) == 0:
            self.errors.append("DataFrame está vazio")
            return
        
        # Verifica duplicatas
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            self.warnings.append(f"Encontradas {duplicate_count} linhas duplicadas")
        
        # Verifica valores negativos em colunas que não deveriam ter
        non_negative_cols = ['Passageiros', 'Distancia', 'Quant Gratuidade'] + self.config.COLS_PAGANTES
        
        for col in non_negative_cols:
            if col in df.columns:
                negative_count = (pd.to_numeric(df[col], errors='coerce') < 0).sum()
                if negative_count > 0:
                    self.warnings.append(
                        f"Coluna '{col}' tem {negative_count} valores negativos"
                    )
        
        # Verifica completude de dados essenciais
        essential_cols = ['Passageiros', 'Nome Linha']
        
        for col in essential_cols:
            if col in df.columns:
                completeness = df[col].notna().sum() / len(df)
                if completeness < 0.8:
                    self.warnings.append(
                        f"Coluna '{col}' tem baixa completude "
                        f"({completeness*100:.1f}% preenchido)"
                    )
    
    def validate_business_rules(self, df: pd.DataFrame):
        """Valida regras de negócio"""
        # Valida duração das viagens
        if all(col in df.columns for col in ['Data Hora Inicio Operacao', 'Data Hora Final Operacao']):
            inicio = pd.to_datetime(df['Data Hora Inicio Operacao'], errors='coerce')
            fim = pd.to_datetime(df['Data Hora Final Operacao'], errors='coerce')
            duracao = (fim - inicio).dt.total_seconds() / 3600
            
            # Viagens muito curtas (< 5 minutos)
            very_short = (duracao < 5/60).sum()
            if very_short > 0:
                self.warnings.append(
                    f"{very_short} viagens com duração menor que 5 minutos"
                )
            
            # Viagens muito longas (> 5 horas)
            very_long = (duracao > 5).sum()
            if very_long > 0:
                self.warnings.append(
                    f"{very_long} viagens com duração maior que 5 horas"
                )
        
        # Valida passageiros vs capacidade típica
        if 'Passageiros' in df.columns:
            pax = pd.to_numeric(df['Passageiros'], errors='coerce')
            
            # Viagens com mais de 200 passageiros (suspeito para ônibus único)
            high_pax = (pax > 200).sum()
            if high_pax > 0:
                self.warnings.append(
                    f"{high_pax} viagens com mais de 200 passageiros"
                )
        
        # Valida proporção gratuidade/pagantes
        if 'Quant Gratuidade' in df.columns and any(col in df.columns for col in self.config.COLS_PAGANTES):
            paying_cols = [col for col in self.config.COLS_PAGANTES if col in df.columns]
            total_paying = df[paying_cols].sum(axis=1)
            gratuity = pd.to_numeric(df['Quant Gratuidade'], errors='coerce')
            
            prop_gratuity = gratuity / (total_paying + gratuity)
            high_gratuity = (prop_gratuity > 0.8).sum()
            
            if high_gratuity > 0:
                self.warnings.append(
                    f"{high_gratuity} viagens com mais de 80% de gratuidades"
                )
    
    def validate_file_structure(self, file_path: str) -> bool:
        """
        Valida estrutura do arquivo antes de carregar
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            True se válido
        """
        import os
        
        # Verifica se arquivo existe
        if not os.path.exists(file_path):
            self.errors.append(f"Arquivo não encontrado: {file_path}")
            return False
        
        # Verifica extensão
        valid_extensions = ['.csv', '.xlsx', '.xls']
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension not in valid_extensions:
            self.errors.append(
                f"Extensão inválida: {extension}. "
                f"Esperadas: {', '.join(valid_extensions)}"
            )
            return False
        
        # Verifica tamanho
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        if file_size > 100:
            self.warnings.append(
                f"Arquivo muito grande ({file_size:.1f} MB). "
                "Pode haver lentidão no processamento."
            )
        
        if file_size == 0:
            self.errors.append("Arquivo está vazio")
            return False
        
        return True
