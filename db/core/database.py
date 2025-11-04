from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import DATABASE_URL_SYNC, DATABASE_URL_ASYNC

Base = declarative_base()

# Engines
engine = create_engine(DATABASE_URL_SYNC, pool_pre_ping=True)
async_engine = create_async_engine(DATABASE_URL_ASYNC, pool_pre_ping=True)

# Session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Helpers
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db():
    async with AsyncSessionLocal() as db:
        yield db

def init_db():
    Base.metadata.create_all(bind=engine)

async def init_async_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)