from pydantic import BaseModel

class QueryLLMMetricBase(BaseModel):
    input_token: int | None = None
    output_token: int | None = None
    thought_token: int | None = None
    total_token: int | None = None

class QueryLLMMetricCreate(QueryLLMMetricBase):
    pass

class QueryLLMMetricUpdate(QueryLLMMetricBase):
    pass

class QueryLLMMetricOut(QueryLLMMetricBase):
    id: int

    class Config:
        orm_mode = True