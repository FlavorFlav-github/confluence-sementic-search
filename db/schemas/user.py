from pydantic import BaseModel

class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    pass

class UserUpdate(BaseModel):
    email: str | None = None

class UserOut(UserBase):
    id: int
    class Config:
        from_attributes = True