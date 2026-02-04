"""
密码重置服务
处理验证码生成、存储、验证和密码重置
"""
import random
import string
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.db.models import User
from src.core.security import get_password_hash, validate_password_strength


class PasswordResetService:
    """密码重置服务类"""

    # 验证码存储（生产环境应使用Redis）
    _verification_codes: Dict[str, Dict[str, Any]] = {}

    # 验证码有效期（分钟）
    CODE_EXPIRE_MINUTES = 15

    def __init__(self, db: AsyncSession):
        self.db = db

    def _generate_code(self, length: int = 6) -> str:
        """生成数字验证码"""
        return ''.join(random.choices(string.digits, k=length))

    def _get_code_key(self, email: str) -> str:
        """获取验证码存储键"""
        return f"reset_code:{email}"

    async def send_verification_code(self, email: str) -> Dict[str, Any]:
        """
        发送验证码到邮箱

        Args:
            email: 用户邮箱

        Returns:
            Dict: 包含成功信息和过期时间的字典

        Raises:
            ValueError: 邮箱不存在或发送过于频繁
        """
        # 检查邮箱是否存在
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        user = result.scalar_one_or_none()

        if user is None:
            # 出于安全考虑，不暴露邮箱是否存在
            # 但可以返回成功消息
            return {
                "success": True,
                "message": "如果该邮箱已注册，验证码将发送至您的邮箱",
                "expires_in": self.CODE_EXPIRE_MINUTES * 60
            }

        # 检查是否频繁发送（1分钟内不能重复发送）
        key = self._get_code_key(email)
        if key in self._verification_codes:
            last_sent = self._verification_codes[key].get("sent_at")
            if last_sent and (datetime.utcnow() - last_sent) < timedelta(minutes=1):
                raise ValueError("验证码发送过于频繁，请1分钟后再试")

        # 生成验证码
        code = self._generate_code()
        expires_at = datetime.utcnow() + timedelta(minutes=self.CODE_EXPIRE_MINUTES)

        # 存储验证码（生产环境应使用Redis）
        self._verification_codes[key] = {
            "code": code,
            "email": email,
            "sent_at": datetime.utcnow(),
            "expires_at": expires_at
        }

        # TODO: 实际发送邮件
        # await self._send_email(email, code)
        # 开发环境下，打印验证码到控制台
        print(f"[Password Reset] Verification code for {email}: {code}")

        return {
            "success": True,
            "message": "如果该邮箱已注册，验证码将发送至您的邮箱",
            "expires_in": self.CODE_EXPIRE_MINUTES * 60,
            # 仅开发环境返回验证码
            "_dev_code": code
        }

    async def verify_code(self, email: str, code: str) -> Dict[str, Any]:
        """
        验证验证码

        Args:
            email: 用户邮箱
            code: 验证码

        Returns:
            Dict: 包含验证结果的字典

        Raises:
            ValueError: 验证码无效或已过期
        """
        key = self._get_code_key(email)

        if key not in self._verification_codes:
            raise ValueError("验证码无效或已过期")

        stored_data = self._verification_codes[key]

        # 检查是否过期
        if datetime.utcnow() > stored_data["expires_at"]:
            del self._verification_codes[key]
            raise ValueError("验证码已过期，请重新获取")

        # 验证码比对
        if stored_data["code"] != code:
            raise ValueError("验证码错误")

        return {
            "success": True,
            "message": "验证码验证成功"
        }

    async def reset_password(
        self,
        email: str,
        code: str,
        new_password: str
    ) -> Dict[str, Any]:
        """
        重置密码

        Args:
            email: 用户邮箱
            code: 验证码
            new_password: 新密码

        Returns:
            Dict: 包含重置结果的字典

        Raises:
            ValueError: 验证码无效或密码不符合要求
        """
        # 先验证验证码
        await self.verify_code(email, code)

        # 验证密码强度
        is_valid, error_msg = validate_password_strength(new_password)
        if not is_valid:
            raise ValueError(error_msg)

        # 获取用户
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        user = result.scalar_one_or_none()

        if user is None:
            raise ValueError("用户不存在")

        # 更新密码
        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.utcnow()

        await self.db.commit()

        # 清除验证码
        key = self._get_code_key(email)
        if key in self._verification_codes:
            del self._verification_codes[key]

        return {
            "success": True,
            "message": "密码重置成功，请使用新密码登录"
        }

    async def _send_email(self, email: str, code: str) -> None:
        """
        发送邮件（需要配置SMTP服务）

        Args:
            email: 收件人邮箱
            code: 验证码
        """
        # TODO: 实现邮件发送
        # 可以使用 smtplib 或第三方服务（如阿里云邮件、SendGrid等）
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # 示例代码（需要配置SMTP服务器）
        # smtp_server = "smtp.example.com"
        # smtp_port = 587
        # smtp_username = "your_username"
        # smtp_password = "your_password"

        # message = MIMEMultipart("alternative")
        # message["Subject"] = "CellMind 密码重置验证码"
        # message["From"] = "noreply@cellmind.com"
        # message["To"] = email

        # html_content = f"""
        # <html>
        # <body>
        #     <h2>CellMind 密码重置</h2>
        #     <p>您正在重置CellMind账户密码。</p>
        #     <p>您的验证码是：<strong>{code}</strong></p>
        #     <p>验证码有效期为15分钟。</p>
        #     <p>如果这不是您的操作，请忽略此邮件。</p>
        # </body>
        # </html>
        # """

        # part = MIMEText(html_content, "html")
        # message.attach(part)

        # with smtplib.SMTP(smtp_server, smtp_port) as server:
        #     server.starttls()
        #     server.login(smtp_username, smtp_password)
        #     server.send_message(message)
        pass
