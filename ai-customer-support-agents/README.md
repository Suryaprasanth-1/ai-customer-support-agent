
# AI Customer Support Agent

LLM-powered customer support agent with intent classification and response routing.

## Features
- Intent detection with confidence scoring
- Safe fallback when confidence is low
- Specialized routing per intent
- FastAPI-based REST API

## Run
pip install -r requirements.txt
uvicorn main:app --reload
