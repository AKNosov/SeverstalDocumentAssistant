import streamlit as st
import os
import json
import tempfile
from pathlib import Path
from document_assistant import DocumentAssistant

st.set_page_config(
    page_title="Document Assistant",
    layout="wide"
)

if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'indexed_files' not in st.session_state:
    st.session_state.indexed_files = []

st.title("Document Assistant")
st.markdown("Интеллектуальный помощник для работы с документами")

with st.sidebar:
    st.header("Настройки")
    
    st.subheader("Параметры чанкирования")
    chunk_size = st.slider("Размер чанка", 100, 2000, 500, 50)
    chunk_overlap = st.slider("Перекрытие", 0, 500, 100, 10)
    
    st.subheader("Параметры поиска")
    top_k = st.slider("Число релевантных фрагментов (K)", 1, 10, 3, 1)
    
    st.subheader("Настройки LLM")
    use_mock = st.checkbox("Использовать mock-режим (без API)", value=True)
    
    if not use_mock:
        ollama_url = st.text_input("Ollama URL", placeholder="Например: http://localhost:11434")
        llm_model = st.text_input("Модель", placeholder="Например: llama3.1")
    
    st.subheader("Модель эмбеддингов")
    embedding_model = st.selectbox(
        "Модель",
        ['paraphrase-multilingual-MiniLM-L12-v2', 'all-mpnet-base-v2']
    )

st.header("Загрузка документов")

uploaded_files = st.file_uploader(
    "Выберите файлы",
    type=['txt', 'docx', 'pdf'],
    accept_multiple_files=True
)

col1, col2 = st.columns([1, 3])

with col1:
    index_button = st.button("Индексировать документы", type="primary", use_container_width=True)

with col2:
    if st.session_state.indexed_files:
        st.success(f"Проиндексировано файлов: {len(st.session_state.indexed_files)} | Чанков: {len(st.session_state.assistant.chunks) if st.session_state.assistant else 0}")

if index_button:
    if not uploaded_files:
        st.error("Загрузите хотя бы один документ")
    else:
        with st.spinner("Инициализация ассистента..."):
            try:
                st.session_state.assistant = DocumentAssistant(
                    embedding_model=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    ollama_url=ollama_url,
                    llm_model=llm_model,
                    use_mock_llm=use_mock
                )
            except Exception as e:
                st.error(f"Ошибка инициализации: {e}")
                st.stop()
        
        temp_paths = []
        for uploaded_file in uploaded_files:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_paths.append(tmp.name)
        
        with st.spinner("Индексация документов..."):
            st.session_state.assistant.index_documents(temp_paths)
            st.session_state.indexed_files = [f.name for f in uploaded_files]
        
        for path in temp_paths:
            try:
                os.remove(path)
            except:
                pass
        
        stats = st.session_state.assistant.get_stats()
        st.success(f"Индексация завершена! Чанков: {stats['total_chunks']}")

st.divider()

st.header("Поисковой запрос")

query = st.text_area("",placeholder="Например: Какие запрещённые действия указаны в пользовательском соглашении?", height=80)

col1, col2 = st.columns([1, 4])

with col1:
    search_button = st.button("Получить ответ", type="primary", use_container_width=True)

if search_button:
    if not st.session_state.assistant:
        st.error("Сначала проиндексируйте документы")
    elif not query.strip():
        st.error("Введите Поисковой запрос")
    else:
        with st.spinner("Поиск релевантных фрагментов..."):
            result = st.session_state.assistant.answer_query(query)
            st.session_state.last_result = result

if st.session_state.last_result:
    result = st.session_state.last_result
    
    st.divider()
    st.header("Результат")
    
    st.markdown("### Ответ:")
    st.markdown(result['answer'])
    
    with st.expander("Использованные фрагменты"):
        for i, chunk in enumerate(result['chunks'], 1):
            st.markdown(f"**Фрагмент {i}** (сходство: {chunk['similarity']:.3f})")
            st.markdown(f"*Источник: {chunk['source']}*")
            st.info(chunk['text'])
    
    with st.expander("Параметры запроса"):
        st.json(result['parameters'])
    
    st.divider()
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("Сохранить в JSON", type="secondary", use_container_width=True):
            output_path = "result.json"
            st.session_state.assistant.save_result(result, output_path)
            
            with open(output_path, 'r', encoding='utf-8') as f:
                st.download_button(
                    label="Скачать JSON",
                    data=f.read(),
                    file_name=f"result_{query[:30].replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )

st.divider()

with st.expander("Справка"):
    st.markdown("""
    **Параметры:**
    - *Размер чанка* — длина текстового блока для индексации
    - *Перекрытие* — количество символов перекрытия между чанками
    - *K* — количество релевантных фрагментов для ответа
    
    **Режимы работы:**
    - *Mock-режим* — генерация ответа без API (демонстрация)
    - *Ollama URL* — реальная генерация через Ollama
    """)
