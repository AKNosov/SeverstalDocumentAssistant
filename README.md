# Severstal Document Assistant

Интеллектуальный помощник для работы с текстовыми документами

## Возможности

- Поддержка форматов: TXT, DOCX, PDF
- Семантический поиск по документам
- Генерация ответов с использованием LLM (OpenAI GPT или mock-режим)
- Настраиваемые параметры чанкирования и поиска
- Веб-интерфейс на Streamlit

## Быстрый старт

### Windows
```bash
run_windows.bat
```

### Linux/Mac
```bash
chmod +x run_linux.sh
./run_linux.sh
```

### Ручная установка
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate.bat  # Windows

pip install -r requirements.txt
streamlit run app.py
```

## Использование

1. Откройте http://localhost:8501 в браузере
2. Настройте параметры в боковой панели
3. Загрузите документы и нажмите "Индексировать"
4. Введите вопрос и получите ответ
5. Сохраните результат в JSON

## API-ключ OpenAI

Для использования реальной LLM введите ключ в интерфейсе приложения.

