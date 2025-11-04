from pydantic import BaseModel

class EntityBase(BaseModel):
    user_id: int | None = None
    ip_address: str | None = None
    browser_agent: str | None = None
    location_country: str | None = None
    location_city: str | None = None

class EntityCreate(EntityBase):
    pass

class EntityUpdate(BaseModel):
    ip_address: str | None = None
    browser_agent: str | None = None
    location_country: str | None = None
    location_city: str | None = None

class EntityOut(EntityBase):
    id: int

    class Config:
        from_attributes = True