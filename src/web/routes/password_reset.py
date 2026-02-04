"""
密码重置路由
处理密码重置相关的API请求
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.web.services.password_reset_service import PasswordResetService


router = APIRouter()


class SendCodeRequest(BaseModel):
    """发送验证码请求"""
    email: EmailStr


class VerifyCodeRequest(BaseModel):
    """验证验证码请求"""
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6, description="6位数字验证码")


class ResetPasswordRequest(BaseModel):
    """重置密码请求"""
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6, description="6位数字验证码")
    new_password: str = Field(..., min_length=8, description="新密码，至少8个字符")


@router.post("/password-reset/send-code", response_model=dict)
async def send_verification_code(
    request: SendCodeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    发送密码重置验证码到邮箱

    发送6位数字验证码到指定邮箱。验证码有效期为15分钟。
    出于安全考虑，无论邮箱是否注册，都会返回成功消息。
    """
    service = PasswordResetService(db)
    try:
        result = await service.send_verification_code(request.email)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/password-reset/verify-code", response_model=dict)
async def verify_code(
    request: VerifyCodeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    验证密码重置验证码

    验证用户输入的验证码是否正确。
    """
    service = PasswordResetService(db)
    try:
        result = await service.verify_code(request.email, request.code)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/password-reset/reset-password", response_model=dict)
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    重置用户密码

    使用验证码和新密码重置用户密码。
    重置成功后，用户需要使用新密码重新登录。
    """
    service = PasswordResetService(db)
    try:
        result = await service.reset_password(
            request.email,
            request.code,
            request.new_password
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
