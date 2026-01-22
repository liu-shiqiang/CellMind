"""
认证服务
处理用户注册、登录、Token验证等核心认证逻辑
"""
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4

from src.db.models import User
from src.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    validate_password_strength,
    decode_token,
)
from src.web.config import settings

ANON_USERNAME_PREFIX = "anon_"
ANON_EMAIL_DOMAIN = "anonymous.local"


def is_anonymous_user(user: User) -> bool:
    """判断是否为匿名用户"""
    return user.username.startswith(ANON_USERNAME_PREFIX) and user.email.endswith(f"@{ANON_EMAIL_DOMAIN}")


class AuthService:
    """认证服务类"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """通过ID获取用户"""
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """通过用户名获取用户"""
        result = await self.db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """通过邮箱获取用户"""
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> User:
        """
        创建新用户

        Args:
            username: 用户名
            email: 邮箱
            password: 明文密码
            full_name: 全名（可选）

        Returns:
            User: 创建的用户对象

        Raises:
            ValueError: 用户名或邮箱已存在，或密码强度不符合要求
        """
        # 验证密码强度
        is_valid, error_msg = validate_password_strength(password)
        if not is_valid:
            raise ValueError(error_msg)

        # 检查用户名是否存在
        existing_user = await self.get_user_by_username(username)
        if existing_user:
            raise ValueError(f"用户名 '{username}' 已被使用")

        # 检查邮箱是否存在
        existing_email = await self.get_user_by_email(email)
        if existing_email:
            raise ValueError(f"邮箱 '{email}' 已被注册")

        # 创建用户
        user = User(
            id=str(uuid4()),
            username=username,
            email=email,
            hashed_password=get_password_hash(password),
            full_name=full_name,
            is_active=True,
            is_verified=False,
            created_at=datetime.utcnow(),
        )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def create_anonymous_user(self) -> User:
        """创建匿名用户"""
        while True:
            username = f"{ANON_USERNAME_PREFIX}{uuid4().hex[:12]}"
            email = f"{username}@{ANON_EMAIL_DOMAIN}"
            existing_user = await self.get_user_by_username(username)
            existing_email = await self.get_user_by_email(email)
            if not existing_user and not existing_email:
                break

        user = User(
            id=str(uuid4()),
            username=username,
            email=email,
            hashed_password=get_password_hash(uuid4().hex),
            full_name=None,
            is_active=True,
            is_verified=False,
            created_at=datetime.utcnow(),
        )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[User]:
        """
        验证用户凭据

        Args:
            username: 用户名或邮箱
            password: 明文密码

        Returns:
            Optional[User]: 验证成功返回用户对象，失败返回 None
        """
        # 尝试用用户名查找，失败则用邮箱查找
        user = await self.get_user_by_username(username)
        if user is None:
            user = await self.get_user_by_email(username)

        if user is None:
            return None

        if not verify_password(password, user.hashed_password):
            return None

        if not user.is_active:
            return None

        # 更新最后登录时间
        user.last_login_at = datetime.utcnow()
        await self.db.commit()

        return user

    def create_tokens(self, user: User) -> Dict[str, Any]:
        """
        为用户创建访问令牌和刷新令牌

        Args:
            user: 用户对象

        Returns:
            Dict: 包含 access_token, refresh_token, expires_in 的字典
        """
        access_token = create_access_token(
            subject=user.id,
            additional_claims={
                "username": user.username,
                "email": user.email,
            }
        )

        refresh_token = create_refresh_token(subject=user.id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }

    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        用户登录

        Args:
            username: 用户名或邮箱
            password: 明文密码

        Returns:
            Dict: 包含 tokens 和 user 信息的字典

        Raises:
            ValueError: 登录失败
        """
        user = await self.authenticate_user(username, password)

        if user is None:
            raise ValueError("用户名或密码错误")

        tokens = self.create_tokens(user)

        return {
            **tokens,
            "user": self._user_to_dict(user),
        }

    async def register(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        用户注册

        Args:
            username: 用户名
            email: 邮箱
            password: 明文密码
            full_name: 全名（可选）

        Returns:
            Dict: 包含 user 信息和 tokens 的字典
        """
        user = await self.create_user(username, email, password, full_name)
        tokens = self.create_tokens(user)

        return {
            **tokens,
            "user": self._user_to_dict(user),
        }

    async def verify_token(self, token: str) -> Optional[User]:
        """
        验证 Token 并返回用户

        Args:
            token: JWT Token

        Returns:
            Optional[User]: Token 有效返回用户对象，无效返回 None
        """
        payload = decode_token(token)
        if payload is None or payload.get("type") != "access":
            return None
        user_id = payload.get("sub")
        if user_id is None:
            return None

        return await self.get_user_by_id(user_id)

    def _user_to_dict(self, user: User) -> Dict[str, Any]:
        """将用户对象转换为字典（不包含敏感信息）"""
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
            "is_anonymous": is_anonymous_user(user),
        }
