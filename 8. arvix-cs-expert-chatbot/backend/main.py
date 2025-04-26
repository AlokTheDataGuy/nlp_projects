from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import chat, papers, users

app = FastAPI(title="arXiv Expert Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(papers.router, prefix="/api/papers", tags=["papers"])
app.include_router(users.router, prefix="/api/users", tags=["users"])

@app.get("/")
async def root():
    return {"message": "Welcome to arXiv Expert Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)