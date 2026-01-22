"""
安全工具模块
包含 JWT Token 生成/验证 和 密码哈希/验证
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import bcrypt

from src.web.config import settings

# JWT 配置
SECRET_KEY = getattr(settings, "JWT_SECRET_KEY", "cellmind-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证明文密码与哈希密码是否匹配

    Args:
        plain_password: 明文密码
        hashed_password: 哈希后的密码

    Returns:
        bool: 密码是否匹配
    """
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """
    对密码进行哈希加密

    Args:
        password: 明文密码

    Returns:
        str: 哈希后的密码
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    创建访问令牌

    Args:
        subject: Token 主体 (通常是 user_id)
        expires_delta: 过期时间增量
        additional_claims: 额外的声明信息

    Returns:
        str: JWT Token 字符串
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }

    if additional_claims:
        to_encode.update(additional_claims)

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(subject: str) -> str:
    """
    创建刷新令牌

    Args:
        subject: Token 主体 (通常是 user_id)

    Returns:
        str: Refresh Token 字符串
    """
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    解码并验证 JWT Token

    Args:
        token: JWT Token 字符串

    Returns:
        Optional[Dict]: 解码后的 payload，验证失败返回 None
    """
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"verify_exp": True}
        )
        return payload
    except JWTError as e:
        # JWTError 包含: ExpiredSignatureError, InvalidTokenError 等
        return None


def get_token_payload(token: str) -> Optional[str]:
    """
    从 Token 中获取用户 ID (subject)

    Args:
        token: JWT Token 字符串

    Returns:
        Optional[str]: 用户 ID，无效 Token 返回 None
    """
    payload = decode_token(token)
    if payload is None:
        return None
    return payload.get("sub")


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    验证密码强度

    Args:
        password: 待验证的密码

    Returns:
        tuple[bool, str]: (是否有效, 错误消息)
    """
    # bcrypt 密码长度限制为 72 字节
    if len(password.encode('utf-8')) > 72:
        return False, "密码长度不能超过72个字符"

    if len(password) < 8:
        return False, "密码长度至少为8位"

    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)

    if not (has_upper and has_lower and has_digit):
        return False, "密码必须包含大写字母、小写字母和数字"

    return True, ""
