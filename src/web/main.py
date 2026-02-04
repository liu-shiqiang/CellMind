"""
CellMind FastAPI 应用入口
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.web.config import settings
from src.web.routes import chat, agent, upload, session, visualization, auth, password_reset, artifacts
from src.web import routes_jobs
from src.db.session import init_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("CellMind API starting...")
    # 初始化数据库
    await init_db()
    print("Database initialized.")
    yield
    # 关闭时清理
    await close_db()
    print("CellMind API shutting down...")


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="CellMind API",
        version="1.0.0",
        description="单细胞生物信息学分析平台",
        lifespan=lifespan,
    )

    # CORS配置
    cors_origins = settings.CORS_ORIGINS
    if isinstance(cors_origins, str):
        cors_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
    app.include_router(password_reset.router, prefix="/api/auth", tags=["Password Reset"])
    app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
    app.include_router(agent.router, prefix="/api/agent", tags=["Agent"])
    app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
    app.include_router(session.router, prefix="/api/sessions", tags=["Session"])
    app.include_router(visualization.router, prefix="/api/visualization", tags=["Visualization"])
    app.include_router(artifacts.router, tags=["Artifacts"])
    app.include_router(routes_jobs.router, prefix="/api", tags=["Jobs"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.web.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
