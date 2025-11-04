from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from db.core.database import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model

    # --- Sync ---
    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        return db.get(self.model, id)

    def get_multi(self, db: Session, skip=0, limit=100) -> List[ModelType]:
        return db.execute(select(self.model).offset(skip).limit(limit)).scalars().all()

    def create(self, db: Session, obj_in: CreateSchemaType) -> ModelType:
        obj = self.model(**obj_in.dict())
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj

    def update(self, db: Session, db_obj: ModelType, obj_in: Union[Dict[str, Any], BaseModel]):
        data = obj_in if isinstance(obj_in, dict) else obj_in.dict(exclude_unset=True)
        for k, v in data.items():
            setattr(db_obj, k, v)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, id: Any):
        obj = db.get(self.model, id)
        if obj:
            db.delete(obj)
            db.commit()
        return obj

    # --- Async ---
    async def aget(self, db: AsyncSession, id: Any):
        return await db.get(self.model, id)

    async def aget_multi(self, db: AsyncSession, skip=0, limit=100):
        res = await db.execute(select(self.model).offset(skip).limit(limit))
        return res.scalars().all()

    async def acreate(self, db: AsyncSession, obj_in: CreateSchemaType):
        obj = self.model(**obj_in.dict())
        db.add(obj)
        await db.commit()
        await db.refresh(obj)
        return obj

    async def aupdate(self, db: AsyncSession, db_obj: ModelType, obj_in: Union[Dict[str, Any], BaseModel]):
        data = obj_in if isinstance(obj_in, dict) else obj_in.dict(exclude_unset=True)
        for k, v in data.items():
            setattr(db_obj, k, v)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj

    async def aremove(self, db: AsyncSession, id: Any):
        obj = await db.get(self.model, id)
        if obj:
            await db.delete(obj)
            await db.commit()
        return obj