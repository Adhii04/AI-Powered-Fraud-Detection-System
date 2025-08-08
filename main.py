from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(title="Echo API Test",
             description="Basic API to test data types",
             version="1.0.0")

# Define the input model with explicit types
class TransactionInput(BaseModel):
    # String fields
    trans_date_trans_time: str
    cc_num: str
    merchant: str
    category: str
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: str
    job: str
    dob: str
    trans_num: str
    
    # Numeric fields
    amt: float
    lat: float
    long: float
    city_pop: int
    unix_time: int
    merch_lat: float
    merch_long: float

# Basic endpoint to echo back the input
@app.post("/echo")
async def echo_input(transaction: TransactionInput) -> Dict[str, Any]:
    # Convert the input model to a dictionary
    data = transaction.dict()
    
    # Add some type information for debugging
    type_info = {
        field: str(type(value)).__repr__()
        for field, value in data.items()
    }
    
    return {
        "received_data": data
    }

# Health check endpoint
@app.get("/")
async def root():
    return {"status": "API is running", "message": "Send POST request to /echo to test data types"}

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)