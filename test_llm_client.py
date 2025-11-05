#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тест LLM клиента"""

from src.llm_client import get_llm_client

print('Инициализация LLM клиента...')
client = get_llm_client()

print('\n[Тест 1] Простой запрос:')
answer = client.generate('Что такое RAG? Ответь кратко в одном предложении.')
print(f'Ответ: {answer}')

print('\n[Тест 2] RAG-режим с контекстом:')
context = 'Инструкция: При ошибке фильтра нужно попробовать обновить настройки.'
query = 'Что делать при ошибке фильтра?'
rag_answer = client.generate_rag_answer(query, context)
print(f'Ответ: {rag_answer}')

print('\n✅ LLM клиент работает!')
