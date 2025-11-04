from pydantic import BaseModel
from datetime import datetime

class QueryBase(BaseModel):
    entity_id: int
    query_timing_id: int | None = None
    query_llm_metric_id: int | None = None
    question: str
    model_used: str
    cache_enabled: bool = False
    cache_used: bool = False

class QueryCreate(QueryBase):
    pass

class QueryUpdate(BaseModel):
    question: str | None = None
    model_used: str | None = None
    cache_enabled: bool | None = None
    cache_used: bool | None = None
    query_timing_id: int | None = None
    query_llm_metric_id: int | None = None

class QueryOut(QueryBase):
    id: int
    date_time: datetime

    class Config:
        from_attributes = True