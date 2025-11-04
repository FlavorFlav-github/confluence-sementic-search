from sqlalchemy import Column, Integer, Float
from db.core.database import Base

class QueryTiming(Base):
    __tablename__ = "query_timings"

    id = Column(Integer, primary_key=True, index=True)
    time_processed = Column(Float)
    time_llm_refine = Column(Float)
    time_sem_search = Column(Float)
    time_llm_ask = Column(Float)