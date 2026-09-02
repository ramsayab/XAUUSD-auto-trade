from pydantic import BaseModel
from datetime import datetime

class HistoryBase(BaseModel):
    order_type: str
    profit: int
    start_time: datetime
    end_time: datetime

class HistoryResponse(HistoryBase):
    id: int
    class Config:
        from_attributes = True
