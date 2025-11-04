from db.models import user, entity, query, query_timing, query_llm_metric
from db.schemas import user as user_schema
from db.crud.base import CRUDBase

crud_registry = {
    "user": CRUDBase[user.User, user_schema.UserCreate, user_schema.UserUpdate](user.User),
    "entity": CRUDBase[entity.Entity, None, None](entity.Entity),
    "query": CRUDBase[query.Query, None, None](query.Query),
    "query_timing": CRUDBase[query_timing.QueryTiming, None, None](query_timing.QueryTiming),
    "query_llm_metric": CRUDBase[query_llm_metric.QueryLLMMetric, None, None](query_llm_metric.QueryLLMMetric),
}