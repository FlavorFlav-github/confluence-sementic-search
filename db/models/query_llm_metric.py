from sqlalchemy import Column, Integer
from db.core.database import Base

class QueryLLMMetric(Base):
    __tablename__ = "query_llm_metrics"

    id = Column(Integer, primary_key=True, index=True)
    input_token = Column(Integer)
    output_token = Column(Integer)
    thought_token = Column(Integer)
    total_token = Column(Integer)