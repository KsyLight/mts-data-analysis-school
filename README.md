# FastAPI Seller & Books Project

Проект демонстрирует работу с FastAPI, SQLAlchemy и JWT-аутентификацией для управления продавцами и книгами.

## Установка и запуск

1. **Клонируйте репозиторий и перейдите в папку проекта**:
   ```bash
   git clone <ссылка-на-репозиторий>
   cd mts-shad-fastapi-project
   ```

2. **Создайте виртуальное окружение**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - **Unix/macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Запустите приложение**:
   ```bash
   uvicorn main:app --reload
   ```
   Приложение будет доступно по адресу:  
   [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Эндпоинты

- **Регистрация продавца**: `POST /api/v1/seller`  
- **Список продавцов**: `GET /api/v1/seller`  
- **Получение данных продавца**: `GET /api/v1/seller/{seller_id}` (требуется JWT)  
- **Обновление продавца**: `PUT /api/v1/seller/{seller_id}` (требуется JWT)  
- **Удаление продавца**: `DELETE /api/v1/seller/{seller_id}` (требуется JWT)  
- **Получение токена**: `POST /api/v1/token`  
- **Создание книги**: `POST /api/v1/books/` (требуется JWT)  
- **Обновление книги**: `PUT /api/v1/books/{book_id}` (требуется JWT)  

## Тесты

Для запуска тестов выполните команду:
```bash
pytest test_main.py
```
Ожидаемый результат:
```
collected 5 items
.....
5 passed in X.XXs
```

## Описание структуры

- **`main.py`**: основной код приложения FastAPI.  
- **`test_main.py`**: набор тестов на pytest.  
- **`test.db`**: база данных SQLite (создаётся автоматически при запуске).  
- **`.venv`**: виртуальное окружение Python (может отсутствовать, если не создавалось локально).  
- **`requirements.txt`**: список зависимостей проекта.
