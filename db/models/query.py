from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.sql import func
from db.core.database import Base

class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    query_timing_id = Column(Integer, ForeignKey("query_timings.id"), nullable=True)
    query_llm_metric_id = Column(Integer, ForeignKey("query_llm_metrics.id"), nullable=True)

    date_time = Column(DateTime(timezone=True), server_default=func.now())
    question = Column(String, nullable=False)
    model_used = Column(String, nullable=False)
    cache_enabled = Column(Boolean, default=False)
    cache_used = Column(Boolean, default=False)