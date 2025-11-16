from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
load_dotenv()
from src.agent import ParseAgent

app = FastAPI()

@app.get("/search")
def search_agent(query: str):
    return {"message": "Hello, World!"}

class ParseRequest(BaseModel):
    input: str

class ParseResponse(BaseModel):
    output: Any

@app.post("/parse", response_model=ParseResponse)
def parse_agent(request: ParseRequest) -> ParseResponse:
    if request.input == "":
        return ParseResponse(output="Input is empty")
    
    result = ParseAgent().parse(request.input)
    return ParseResponse(output={"title": result.title, "content": result.content})

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
