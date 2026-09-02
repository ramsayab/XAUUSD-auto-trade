from sqlalchemy import Column, Integer, String, DateTime
from .database import Base

class History(Base):
    __tablename__ = "history_trades"

    id = Column(Integer, primary_key=True, index=True)
    order_type = Column(String(10), nullable=True)
    profit = Column(Integer, nullable=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
