from fastapi import FastAPI
from src.web.api import router

app = FastAPI(
    title="Cell Annotation Agent API",
    version="0.1.0",
)

app.include_router(router, prefix="/api")
