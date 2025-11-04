from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from db.core.database import Base

class Entity(Base):
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    ip_address = Column(String, nullable=True)
    browser_agent = Column(String, nullable=True)
    location_country = Column(String, nullable=True)
    location_city = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint('user_id', 'ip_address', name='uq_user_ip'),
    )