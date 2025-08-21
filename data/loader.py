# -*- coding: utf-8 -*-
"""Módulo de carregamento de dados"""


import pandas as pd
import streamlit as st
from typing import Optional, Union, List
import io


class DataLoader:
    """Classe responsável pelo carregamento de dados"""
    
    ENCODINGS = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    
    def load_csv(self, 
                 file: Union[str, io.BytesIO],
                 separator: str = ";") -> Optional[pd.DataFrame]:
        """
        Carrega arquivo CSV com tratamento de encoding
        
        Args:
            file: Caminho ou objeto de arquivo
            separator: Separador do CSV
            
        Returns:
            DataFrame ou None se falhar
        """
        last_error = None
        
        for encoding in self.ENCODINGS:
            try:
                # Reset file pointer if it's a file object
                if hasattr(file, 'seek'):
                    file.seek(0)
                
                df = pd.read_csv(
                    file,
                    sep=separator,
                    encoding=encoding,
                    low_memory=False
                )
                
                # Limpa espaços nas colunas
                df.columns = [col.strip() for col in df.columns]
                
                return df
                
            except Exception as e:
                last_error = e
                continue
        
        st.error(f"Falha ao ler o arquivo CSV. Erro: {last_error}")
        return None
    
    def load_excel(self,
                   file: Union[str, io.BytesIO],
                   sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Carrega arquivo Excel
        
        Args:
            file: Caminho ou objeto de arquivo
            sheet_name: Nome da planilha (None para primeira)
            
        Returns:
            DataFrame ou None se falhar
        """
        try:
            if hasattr(file, 'seek'):
                file.seek(0)
            
            df = pd.read_excel(
                file,
                sheet_name=sheet_name or 0,
                engine='openpyxl'
            )
            
            # Limpa espaços nas colunas
            df.columns = [col.strip() for col in df.columns]
            
            return df
            
        except Exception as e:
            st.error(f"Falha ao ler o arquivo Excel. Erro: {e}")
            return None
    
    def validate_columns(self,
                        df: pd.DataFrame,
                        required_columns: List[str]) -> tuple[bool, List[str]]:
        """
        Valida se o DataFrame possui as colunas necessárias
        
        Args:
            df: DataFrame a validar
            required_columns: Lista de colunas obrigatórias
            
        Returns:
            Tupla (válido, colunas_faltantes)
        """
        missing = [col for col in required_columns if col not in df.columns]
        return len(missing) == 0, missing