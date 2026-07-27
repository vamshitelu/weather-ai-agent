# Weather AI Agent (Langchain)

A small weather chatbot. The backend is a LangGraph agent(Groq + a live weather tool via open-Meteo, no weeather API Key needed)

## Backup setup

```bash
python -m venv venv
source venv/bin/activate # venv\Scripts\activate on Windows
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY

uvicorn app:app --release --port 8000
```

Test it directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the weather is Hyderabad?"}'
```


