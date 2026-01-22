"""
认证相关 API 路由
处理用户注册、登录、登出、获取当前用户等
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from src.db.session import get_db
from src.web.services.auth_service import AuthService

router = APIRouter()
security = HTTPBearer()


# ==================== Request/Response Schemas ====================

class RegisterRequest(BaseModel):
    """用户注册请求"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱")
    password: str = Field(..., min_length=8, max_length=100, description="密码")
    full_name: Optional[str] = Field(None, max_length=100, description="全名")

    @validator('username')
    def validate_username(cls, v):
        """验证用户名格式"""
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('用户名只能包含字母、数字和下划线')
        return v


class LoginRequest(BaseModel):
    """用户登录请求"""
    username: str = Field(..., description="用户名或邮箱")
    password: str = Field(..., description="密码")


class TokenResponse(BaseModel):
    """Token 响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    """用户信息响应"""
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: Optional[str] = None
    last_login_at: Optional[str] = None
    is_anonymous: bool = False


class AuthResponse(TokenResponse):
    """认证响应（包含用户信息）"""
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """刷新 Token 请求"""
    refresh_token: str


# ==================== Helper Functions ====================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    """
    依赖注入：获取当前登录用户

    Raises:
        HTTPException: Token 无效或用户不存在
    """
    token = credentials.credentials
    auth_service = AuthService(db)
    user = await auth_service.verify_token(token)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UserResponse(**auth_service._user_to_dict(user))


async def get_current_active_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """
    依赖注入：获取当前活跃用户

    Raises:
        HTTPException: 用户账号已被禁用
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户账号已被禁用"
        )
    return current_user


# ==================== Route Handlers ====================

@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    用户注册

    - **username**: 用户名 (3-50字符，只能包含字母数字下划线)
    - **email**: 邮箱地址
    - **password**: 密码 (至少8位，包含大小写字母和数字)
    - **full_name**: 全名 (可选)
    """
    auth_service = AuthService(db)

    try:
        result = await auth_service.register(
            username=request.username,
            email=request.email,
            password=request.password,
            full_name=request.full_name,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    用户登录

    - **username**: 用户名或邮箱
    - **password**: 密码
    """
    auth_service = AuthService(db)

    try:
        result = await auth_service.login(
            username=request.username,
            password=request.password,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/guest", response_model=AuthResponse)
async def guest_login(
    db: AsyncSession = Depends(get_db)
):
    """
    匿名登录（临时用户）
    """
    auth_service = AuthService(db)
    user = await auth_service.create_anonymous_user()
    tokens = auth_service.create_tokens(user)
    return {
        **tokens,
        "user": auth_service._user_to_dict(user),
    }


@router.post("/logout")
async def logout():
    """
    用户登出

    注：由于使用 JWT，登出主要在前端处理（删除本地存储的 Token）
    后端可在此添加 Token 黑名单功能
    """
    return {"message": "登出成功"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    获取当前登录用户信息
    """
    return current_user


@router.post("/verify", response_model=UserResponse)
async def verify_token(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    验证 Token 有效性
    """
    return current_user


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    刷新访问令牌

    - **refresh_token**: 刷新令牌
    """
    from src.core.security import decode_token

    auth_service = AuthService(db)

    # 解码 refresh token
    payload = decode_token(request.refresh_token)
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的刷新令牌"
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的刷新令牌"
        )

    user = await auth_service.get_user_by_id(user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已被禁用"
        )

    tokens = auth_service.create_tokens(user)
    return tokens
