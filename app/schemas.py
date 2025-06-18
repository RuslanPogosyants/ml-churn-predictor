from pydantic import BaseModel


class CustomerData(BaseModel):
    age: int
    gender: str
    income: float
    contract_type: str
