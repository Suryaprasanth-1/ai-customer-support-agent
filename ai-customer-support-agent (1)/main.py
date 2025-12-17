
import os, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="AI Customer Support Agent")

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
)

MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")

class Message(BaseModel):
    text: str

CLASSIFIER_PROMPT = '''
Classify the user query into one intent:
billing | technical | refund | account | sales | other

Return ONLY valid JSON:
{
  "intent": "<intent>",
  "confidence": 0.0
}
'''

ROUTING_PROMPTS = {
    "billing": "You are a billing support agent. Ask for invoice or order ID if required.",
    "technical": "You are technical support. Ask for device, app version, and steps to reproduce.",
    "refund": "You handle refunds. Ask for order ID and explain refund policy.",
    "account": "You handle account issues such as login, verification, or password reset.",
    "sales": "You are a sales agent. Ask clarifying questions and suggest suitable plans.",
    "other": "You are a general support agent. Ask clarifying questions."
}

@app.post("/support")
def support(msg: Message):
    if not os.getenv("LLM_API_KEY"):
        raise HTTPException(500, "LLM_API_KEY not set")

    cls = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": msg.text}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )

    result = json.loads(cls.choices[0].message.content)
    intent = result.get("intent", "other")
    confidence = result.get("confidence", 0.0)

    if confidence < 0.65:
        return {
            "intent": intent,
            "confidence": confidence,
            "response": "Could you please provide a bit more detail so I can assist you better?"
        }

    routed_prompt = ROUTING_PROMPTS.get(intent, ROUTING_PROMPTS["other"])

    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": routed_prompt},
            {"role": "user", "content": msg.text}
        ],
        temperature=0.3
    )

    return {
        "intent": intent,
        "confidence": confidence,
        "response": reply.choices[0].message.content
    }
