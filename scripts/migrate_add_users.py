"""
数据库迁移脚本
添加 users 表用于用户认证功能
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.db.session import engine


async def migrate_add_users_table():
    """添加 users 表"""

    # 1. 创建 users 表
    create_users_sql = """
    CREATE TABLE IF NOT EXISTS users (
        id VARCHAR(36) PRIMARY KEY,
        username VARCHAR(50) UNIQUE NOT NULL,
        email VARCHAR(100) UNIQUE NOT NULL,
        hashed_password VARCHAR(255) NOT NULL,
        full_name VARCHAR(100),
        is_active BOOLEAN DEFAULT TRUE,
        is_verified BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login_at TIMESTAMP
    );
    """

    async with engine.begin() as conn:
        await conn.execute(text(create_users_sql))
        print("✓ Users table created successfully")

    # 2. 创建 username 索引
    create_username_index = """
    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
    """
    async with engine.begin() as conn:
        await conn.execute(text(create_username_index))
        print("✓ Username index created")

    # 3. 创建 email 索引
    create_email_index = """
    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    """
    async with engine.begin() as conn:
        await conn.execute(text(create_email_index))
        print("✓ Email index created")

    print("✓ Migration completed successfully")


if __name__ == "__main__":
    asyncio.run(migrate_add_users_table())
