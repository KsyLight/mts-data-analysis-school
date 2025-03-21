import pytest
from fastapi.testclient import TestClient
from main import app, SessionLocal, Base, engine

# Пересоздаем базу данных перед запуском тестов
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def test_create_seller():
    seller_data = {
        "id": 1,
        "first_name": "Иван",
        "last_name": "Иванов",
        "e_mail": "ivan@example.com",
        "password": "secret"
    }
    response = client.post("/api/v1/seller", json=seller_data)
    assert response.status_code == 200
    data = response.json()
    # Поле password не должно возвращаться
    assert "password" not in data
    assert data["first_name"] == "Иван"

def test_get_sellers():
    response = client.get("/api/v1/seller")
    assert response.status_code == 200
    data = response.json()
    for seller in data:
        assert "password" not in seller

def test_token_and_get_seller():
    seller_data = {
        "id": 2,
        "first_name": "Петр",
        "last_name": "Петров",
        "e_mail": "petr@example.com",
        "password": "mypassword"
    }
    client.post("/api/v1/seller", json=seller_data)
    token_response = client.post(
        "/api/v1/token",
        data={"username": "petr@example.com", "password": "mypassword"}
    )
    assert token_response.status_code == 200
    token = token_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/v1/seller/2", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "seller" in data
    assert "books" in data
    assert "password" not in data["seller"]

def test_update_seller():
    seller_data = {
        "id": 3,
        "first_name": "Сергей",
        "last_name": "Сергеев",
        "e_mail": "sergey@example.com",
        "password": "pass123"
    }
    client.post("/api/v1/seller", json=seller_data)
    token_response = client.post(
        "/api/v1/token",
        data={"username": "sergey@example.com", "password": "pass123"}
    )
    token = token_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    update_data = {
        "first_name": "Сергей",
        "last_name": "Новиков",
        "e_mail": "sergey_new@example.com"
    }
    response = client.put("/api/v1/seller/3", json=update_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["last_name"] == "Новиков"
    assert data["e_mail"] == "sergey_new@example.com"
    assert "password" not in data

def test_delete_seller():
    seller_data = {
        "id": 4,
        "first_name": "Анна",
        "last_name": "Антонова",
        "e_mail": "anna@example.com",
        "password": "anna_secret"
    }
    client.post("/api/v1/seller", json=seller_data)
    token_response = client.post(
        "/api/v1/token",
        data={"username": "anna@example.com", "password": "anna_secret"}
    )
    token = token_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    # Создаем книгу для продавца с id=4
    book_data = {
        "id": 1,
        "title": "Книга Анны",
        "seller_id": 4
    }
    client.post("/api/v1/books/", json=book_data, headers=headers)
    # Удаляем продавца
    response = client.delete("/api/v1/seller/4", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["detail"] == "Seller and associated books deleted"
    # После удаления токен становится невалидным, поэтому запрос возвращает 401 Unauthorized
    response = client.get("/api/v1/seller/4", headers=headers)
    assert response.status_code == 401