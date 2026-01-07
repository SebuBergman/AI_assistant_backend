# app/services/chat_service.py
import json
import httpx
from datetime import datetime
from typing import Optional, List, Dict, Any
from database import get_db_pool, get_redis_client, CACHE_KEYS, CACHE_TTL
import os

class ChatService:
    """Service for managing chats and messages"""

    @staticmethod
    async def create_chat(user_id: str, first_message: str) -> Dict[str, Any]:
        """
        Create a new chat with smart title generation
        """
        pool = await get_db_pool()
        redis = await get_redis_client()

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Generate fallback title
                fallback_title = ChatService._generate_fallback_title(first_message)

                # Insert chat with fallback title
                chat_row = await conn.fetchrow(
                    """
                    INSERT INTO chats (user_id, title)
                    VALUES ($1, $2)
                    RETURNING id, user_id, title, created_at, updated_at
                    """,
                    user_id, fallback_title
                )

                chat = ChatService._map_row_to_chat(chat_row)

                # Add first message
                await conn.execute(
                    """
                    INSERT INTO messages (chat_id, role, content)
                    VALUES ($1, $2, $3)
                    """,
                    chat["id"], "user", first_message
                )

        # Invalidate cache
        await redis.delete(CACHE_KEYS["user_chats"](user_id))

        # Generate better title asynchronously (fire and forget)
        import asyncio
        asyncio.create_task(
            ChatService._generate_and_update_title(
                chat["id"], first_message, user_id
            )
        )

        return chat

    @staticmethod
    async def _generate_and_update_title(
        chat_id: str, message: str, user_id: str
    ) -> None:
        """Generate smart title using your title generation endpoint"""
        try:
            api_url = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{api_url}/chat/title",
                    json={
                        "message": message,
                        "chat_id": chat_id
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                title = data.get("title")

            # Validate title
            if not title or len(title) > 100:
                print("Generated title invalid, keeping fallback")
                return

            # Update chat with generated title
            pool = await get_db_pool()
            redis = await get_redis_client()

            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE chats SET title = $1 WHERE id = $2",
                    title, chat_id
                )

            # Invalidate cache
            await redis.delete(CACHE_KEYS["user_chats"](user_id))

        except Exception as e:
            print(f"Failed to generate/update chat title: {e}")
            # Fail silently - fallback title is in place

    @staticmethod
    def _generate_fallback_title(message: str) -> str:
        """Generate a fallback title from the message"""
        import re
        
        title = message.strip()
        title = re.sub(r'\s+', ' ', title)

        # Try to get first sentence
        match = re.match(r'^[^.!?]+[.!?]', title)
        if match:
            first_sentence = match.group(0)
            if len(first_sentence) < len(title):
                title = first_sentence

        # Truncate to 60 characters
        if len(title) > 60:
            title = title[:57] + "..."

        return title

    @staticmethod
    async def get_user_chats(user_id: str) -> List[Dict[str, Any]]:
        """Get all chats for a user (with caching)"""
        redis = await get_redis_client()
        cache_key = CACHE_KEYS["user_chats"](user_id)

        # Try cache first
        try:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis error: {e}")

        # Query database
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM chats
                WHERE user_id = $1
                ORDER BY updated_at DESC
                LIMIT 50
                """,
                user_id
            )

        chats = [ChatService._map_row_to_chat(row) for row in rows]

        # Cache the result
        try:
            await redis.setex(
                cache_key,
                CACHE_TTL["chats"],
                json.dumps(chats, default=str)
            )
        except Exception as e:
            print(f"Redis cache error: {e}")

        return chats

    @staticmethod
    async def get_chat_by_id(
        chat_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get chat by ID"""
        redis = await get_redis_client()
        cache_key = CACHE_KEYS["chat_meta"](chat_id)

        # Try cache
        try:
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis error: {e}")

        # Query database
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM chats
                WHERE id = $1 AND user_id = $2
                """,
                chat_id, user_id
            )

        if not row:
            return None

        chat = ChatService._map_row_to_chat(row)

        # Cache
        try:
            await redis.setex(
                cache_key,
                CACHE_TTL["chats"],
                json.dumps(chat, default=str)
            )
        except Exception as e:
            print(f"Redis cache error: {e}")

        return chat

    @staticmethod
    async def get_chat_messages(chat_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a chat (with caching)"""
        print(f"ChatService.get_chat_messages called for chatId: {chat_id}")
        redis = await get_redis_client()
        cache_key = CACHE_KEYS["chat_messages"](chat_id)
        print(f"Using cacheKey: {cache_key}")

        # Try cache
        try:
            cached = await redis.get(cache_key)
            if cached:
                print(f"Cache hit for chatId: {chat_id}")
                return json.loads(cached)
        except Exception as e:
            print(f"Redis error: {e}")

        print(f"Cache miss for chatId: {chat_id}")

        # Query database
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM messages
                WHERE chat_id = $1
                ORDER BY created_at ASC
                """,
                chat_id
            )

        print(f"Fetched messages from DB for chatId: {chat_id}, Rows: {len(rows)}")

        messages = [ChatService._map_row_to_message(row) for row in rows]
        print(f"Mapped messages for chatId: {chat_id}, Messages: {messages}")

        # Cache
        try:
            await redis.setex(
                cache_key,
                CACHE_TTL["messages"],
                json.dumps(messages, default=str)
            )
        except Exception as e:
            print(f"Redis cache error: {e}")

        print(f"Returning messages for chatId: {chat_id}, Messages: {messages}")
        return messages

    @staticmethod
    async def add_message(
        chat_id: str,
        role: str,
        content: str,
        references: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Add a message to a chat"""
        pool = await get_db_pool()
        redis = await get_redis_client()

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Insert message with references
                row = await conn.fetchrow(
                    """
                    INSERT INTO messages (chat_id, role, content, rag_references)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id, chat_id, role, content, created_at, rag_references
                    """,
                    chat_id,
                    role,
                    content,
                    json.dumps(references) if references else None
                )

                # Update chat timestamp
                await conn.execute(
                    "UPDATE chats SET updated_at = NOW() WHERE id = $1",
                    chat_id
                )

        message = ChatService._map_row_to_message(row)

        # Invalidate caches
        await redis.delete(CACHE_KEYS["chat_messages"](chat_id))
        await redis.delete(CACHE_KEYS["chat_meta"](chat_id))

        return message

    @staticmethod
    async def delete_chat(chat_id: str, user_id: str) -> bool:
        """Delete a chat"""
        redis = await get_redis_client()
        pool = await get_db_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chats WHERE id = $1 AND user_id = $2",
                chat_id, user_id
            )

        # Clear caches
        await redis.delete(CACHE_KEYS["chat_messages"](chat_id))
        await redis.delete(CACHE_KEYS["chat_meta"](chat_id))
        await redis.delete(CACHE_KEYS["user_chats"](user_id))

        return result != "DELETE 0"

    @staticmethod
    async def delete_all_chats(user_id: str) -> int:
        """Delete all chats for a user"""
        redis = await get_redis_client()
        pool = await get_db_pool()

        async with pool.acquire() as conn:
            # Get all chat IDs first
            rows = await conn.fetch(
                "SELECT id FROM chats WHERE user_id = $1",
                user_id
            )

            # Delete all chats
            result = await conn.execute(
                "DELETE FROM chats WHERE user_id = $1",
                user_id
            )

        # Clear caches for each chat
        cache_deletions = []
        for row in rows:
            cache_deletions.append(redis.delete(CACHE_KEYS["chat_messages"](row["id"])))
            cache_deletions.append(redis.delete(CACHE_KEYS["chat_meta"](row["id"])))

        cache_deletions.append(redis.delete(CACHE_KEYS["user_chats"](user_id)))

        import asyncio
        await asyncio.gather(*cache_deletions)

        # Extract count from result string "DELETE N"
        count = int(result.split()[-1]) if result.split() else 0
        return count

    @staticmethod
    async def update_chat_title(
        chat_id: str, user_id: str, title: str
    ) -> bool:
        """Update chat title"""
        redis = await get_redis_client()
        pool = await get_db_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE chats
                SET title = $1, updated_at = NOW()
                WHERE id = $2 AND user_id = $3
                """,
                title, chat_id, user_id
            )

        # Clear caches
        await redis.delete(CACHE_KEYS["chat_meta"](chat_id))
        await redis.delete(CACHE_KEYS["user_chats"](user_id))

        return result != "UPDATE 0"

    # Helper methods
    @staticmethod
    def _map_row_to_chat(row) -> Dict[str, Any]:
        """Map database row to chat dictionary"""
        return {
            "id": row["id"],
            "userId": row["user_id"],
            "title": row["title"],
            "createdAt": row["created_at"].isoformat() if isinstance(row["created_at"], datetime) else row["created_at"],
            "updatedAt": row["updated_at"].isoformat() if isinstance(row["updated_at"], datetime) else row["updated_at"],
        }

    @staticmethod
    def _map_row_to_message(row) -> Dict[str, Any]:
        """Map database row to message dictionary"""
        rag_refs = None
        if row.get("rag_references"):
            if isinstance(row["rag_references"], str):
                rag_refs = json.loads(row["rag_references"])
            elif isinstance(row["rag_references"], list):
                rag_refs = row["rag_references"]

        message = {
            "id": row["id"],
            "chatId": row["chat_id"],
            "role": row["role"],
            "content": row["content"],
            "createdAt": row["created_at"].isoformat() if isinstance(row["created_at"], datetime) else row["created_at"],
        }

        if rag_refs:
            message["rag_references"] = rag_refs

        return message

    @staticmethod
    async def test_connection():
        """Test database connection"""
        try:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchrow("SELECT NOW()")
                print(f"Database connected: {result}")
                return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False