# -*- coding: utf-8 -*-
"""Utilitários de formatação para padrão brasileiro"""


from typing import Union, Optional


class BrazilianFormatter:
    """Formatador para padrão brasileiro (PT-BR)"""
    
    @staticmethod
    def format_number(value: Union[int, float], 
                     decimals: int = 2) -> str:
        """
        Formata número no padrão brasileiro
        
        Args:
            value: Valor a formatar
            decimals: Número de casas decimais
            
        Returns:
            String formatada
        """
        try:
            if decimals == 0:
                return f"{int(round(float(value))):,}".replace(",", ".")
            
            formatted = f"{float(value):,.{decimals}f}"
            # Substitui vírgula por X, ponto por vírgula, X por ponto
            formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
            return formatted
            
        except (ValueError, TypeError):
            return "0" if decimals == 0 else "0," + "0" * decimals
    
    @staticmethod
    def format_currency(value: Union[int, float],
                       decimals: int = 2) -> str:
        """
        Formata valor monetário em Reais
        
        Args:
            value: Valor monetário
            decimals: Número de casas decimais
            
        Returns:
            String formatada com R$
        """
        formatted_number = BrazilianFormatter.format_number(value, decimals)
        return f"R$ {formatted_number}"
    
    @staticmethod
    def format_percentage(value: Union[int, float],
                         decimals: int = 1) -> str:
        """
        Formata percentual
        
        Args:
            value: Valor (0-1 para percentual, ou já em percentual)
            decimals: Número de casas decimais
            
        Returns:
            String formatada com %
        """
        # Se valor menor que 1, assume que está em decimal
        if abs(value) <= 1:
            value = value * 100
        
        formatted_number = BrazilianFormatter.format_number(value, decimals)
        return f"{formatted_number}%"
    
    @staticmethod
    def format_time_hhmm(minutes: Union[int, float]) -> str:
        """
        Formata minutos em HH:MM
        
        Args:
            minutes: Total de minutos
            
        Returns:
            String no formato HH:MM
        """
        try:
            total_minutes = int(round(float(minutes)))
            hours = total_minutes // 60
            mins = total_minutes % 60
            return f"{hours:02d}:{mins:02d}"
        except (ValueError, TypeError):
            return "00:00"
    
    @staticmethod
    def format_integer(value: Union[int, float]) -> str:
        """
        Formata número inteiro com separador de milhares
        
        Args:
            value: Valor inteiro
            
        Returns:
            String formatada
        """
        return BrazilianFormatter.format_number(value, 0)