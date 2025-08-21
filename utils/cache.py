# -*- coding: utf-8 -*-
"""Gerenciador de cache"""


import streamlit as st
import hashlib
import pickle
from typing import Any, Optional
from datetime import datetime, timedelta


class CacheManager:
    """Gerenciador de cache para o dashboard"""
    
    def __init__(self):
        self.init_cache()
    
    def init_cache(self):
        """Inicializa estrutura de cache no session state"""
        if 'cache' not in st.session_state:
            st.session_state.cache = {}
        
        if 'cache_timestamps' not in st.session_state:
            st.session_state.cache_timestamps = {}
    
    def get_cache_key(self, *args) -> str:
        """
        Gera chave de cache baseada nos argumentos
        
        Args:
            *args: Argumentos para gerar a chave
            
        Returns:
            Chave de cache (hash)
        """
        # Serializa argumentos
        serialized = pickle.dumps(args)
        
        # Gera hash
        return hashlib.md5(serialized).hexdigest()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Recupera valor do cache
        
        Args:
            key: Chave do cache
            default: Valor padrão se não encontrado
            
        Returns:
            Valor em cache ou padrão
        """
        return st.session_state.cache.get(key, default)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Armazena valor no cache
        
        Args:
            key: Chave do cache
            value: Valor a armazenar
            ttl: Tempo de vida em segundos (None = permanente)
        """
        st.session_state.cache[key] = value
        
        if ttl:
            expiry = datetime.now() + timedelta(seconds=ttl)
            st.session_state.cache_timestamps[key] = expiry
    
    def is_expired(self, key: str) -> bool:
        """
        Verifica se cache expirou
        
        Args:
            key: Chave do cache
            
        Returns:
            True se expirou ou não existe
        """
        if key not in st.session_state.cache:
            return True
        
        if key in st.session_state.cache_timestamps:
            return datetime.now() > st.session_state.cache_timestamps[key]
        
        return False
    
    def clear(self, key: Optional[str] = None):
        """
        Limpa cache
        
        Args:
            key: Chave específica ou None para limpar tudo
        """
        if key:
            st.session_state.cache.pop(key, None)
            st.session_state.cache_timestamps.pop(key, None)
        else:
            st.session_state.cache.clear()
            st.session_state.cache_timestamps.clear()
    
    def cached_function(self, ttl: Optional[int] = None):
        """
        Decorator para cachear resultado de função
        
        Args:
            ttl: Tempo de vida em segundos
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Gera chave incluindo nome da função
                cache_key = self.get_cache_key(func.__name__, args, kwargs)
                
                # Verifica se existe e não expirou
                if not self.is_expired(cache_key):
                    return self.get(cache_key)
                
                # Executa função
                result = func(*args, **kwargs)
                
                # Armazena no cache
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator