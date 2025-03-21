from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import jwt

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base

# --- Конфигурация БД ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"  # База SQLite в файле
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- SQLAlchemy модели ---
class SellerModel(Base):
    __tablename__ = "sellers"
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    e_mail = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    # При удалении продавца автоматически удаляются все связанные книги
    books = relationship("BookModel", back_populates="seller", cascade="all, delete")

class BookModel(Base):
    __tablename__ = "books"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    seller_id = Column(Integer, ForeignKey("sellers.id"), nullable=False)
    seller = relationship("SellerModel", back_populates="books")

Base.metadata.create_all(bind=engine)

# --- Pydantic схемы ---
class SellerCreate(BaseModel):
    id: int
    first_name: str
    last_name: str
    e_mail: str
    password: str

class SellerOut(BaseModel):
    id: int
    first_name: str
    last_name: str
    e_mail: str

    model_config = ConfigDict(from_attributes=True)

class SellerUpdate(BaseModel):
    first_name: str
    last_name: str
    e_mail: str

class BookCreate(BaseModel):
    id: int
    title: str
    seller_id: int

class BookOut(BaseModel):
    id: int
    title: str
    seller_id: int

    model_config = ConfigDict(from_attributes=True)

class SellerDetail(BaseModel):
    seller: SellerOut
    books: List[BookOut]

# --- JWT настройки ---
JWT_SECRET = "your_jwt_secret"  # Замените на ваш секретный ключ
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

# --- FastAPI приложение и зависимости ---
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_token(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        seller_id = int(payload.get("sub"))
        if seller_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        seller = db.query(SellerModel).filter(SellerModel.id == seller_id).first()
        if seller is None:
            raise HTTPException(status_code=401, detail="Seller not found")
        return seller
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- Эндпоинты приложения ---

# 1. Регистрация продавца
@app.post("/api/v1/seller", response_model=SellerOut)
def create_seller(seller: SellerCreate, db: Session = Depends(get_db)):
    existing = db.query(SellerModel).filter(
        (SellerModel.id == seller.id) | (SellerModel.e_mail == seller.e_mail)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Seller with such id or e_mail already exists")
    new_seller = SellerModel(**seller.model_dump())
    db.add(new_seller)
    db.commit()
    db.refresh(new_seller)
    return new_seller

# 2. Получение списка продавцов (без поля password)
@app.get("/api/v1/seller", response_model=List[SellerOut])
def get_sellers(db: Session = Depends(get_db)):
    sellers = db.query(SellerModel).all()
    return sellers

# 3. Получение данных продавца с книгами (защищён JWT)
@app.get("/api/v1/seller/{seller_id}", response_model=SellerDetail)
def get_seller(seller_id: int, token_seller: SellerModel = Depends(verify_token), db: Session = Depends(get_db)):
    seller = db.query(SellerModel).filter(SellerModel.id == seller_id).first()
    if not seller:
        raise HTTPException(status_code=404, detail="Seller not found")
    return {"seller": seller, "books": seller.books}

# 4. Обновление данных продавца (без изменения пароля и книг) – защищён JWT
@app.put("/api/v1/seller/{seller_id}", response_model=SellerOut)
def update_seller(seller_id: int, updated_data: SellerUpdate, token_seller: SellerModel = Depends(verify_token), db: Session = Depends(get_db)):
    seller = db.query(SellerModel).filter(SellerModel.id == seller_id).first()
    if not seller:
        raise HTTPException(status_code=404, detail="Seller not found")
    seller.first_name = updated_data.first_name
    seller.last_name = updated_data.last_name
    seller.e_mail = updated_data.e_mail
    db.commit()
    db.refresh(seller)
    return seller

# 5. Удаление продавца (и связанных книг) – защищён JWT
@app.delete("/api/v1/seller/{seller_id}")
def delete_seller(seller_id: int, token_seller: SellerModel = Depends(verify_token), db: Session = Depends(get_db)):
    seller = db.query(SellerModel).filter(SellerModel.id == seller_id).first()
    if not seller:
        raise HTTPException(status_code=404, detail="Seller not found")
    db.delete(seller)
    db.commit()
    return {"detail": "Seller and associated books deleted"}

# 6. Получение JWT токена по e_mail и password
@app.post("/api/v1/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    seller = db.query(SellerModel).filter(
        SellerModel.e_mail == form_data.username,
        SellerModel.password == form_data.password
    ).first()
    if not seller:
        raise HTTPException(status_code=400, detail="Invalid e_mail or password")
    access_token = create_access_token(data={"sub": str(seller.id)})
    return {"access_token": access_token, "token_type": "bearer"}

# 7. Создание книги – защищён JWT
@app.post("/api/v1/books/", response_model=BookOut)
def create_book(book: BookCreate, token_seller: SellerModel = Depends(verify_token), db: Session = Depends(get_db)):
    existing = db.query(BookModel).filter(BookModel.id == book.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Book with such id already exists")
    new_book = BookModel(**book.model_dump())
    db.add(new_book)
    db.commit()
    db.refresh(new_book)
    return new_book

# 8. Обновление книги – защищён JWT
@app.put("/api/v1/books/{book_id}", response_model=BookOut)
def update_book(book_id: int, updated_book: BookCreate, token_seller: SellerModel = Depends(verify_token), db: Session = Depends(get_db)):
    book = db.query(BookModel).filter(BookModel.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    book.title = updated_book.title
    book.seller_id = updated_book.seller_id
    db.commit()
    db.refresh(book)
    return book

# Корневой эндпоинт для проверки
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Для запуска приложения:
# uvicorn main:app --reload