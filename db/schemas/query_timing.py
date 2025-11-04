from pydantic import BaseModel

class QueryTimingBase(BaseModel):
    time_processed: float | None = None
    time_llm_refine: float | None = None
    time_sem_search: float | None = None
    time_llm_ask: float | None = None

class QueryTimingCreate(QueryTimingBase):
    pass

class QueryTimingUpdate(QueryTimingBase):
    pass

class QueryTimingOut(QueryTimingBase):
    id: int

    class Config:
        from_attributes = True