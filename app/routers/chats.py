# app/routers/chats.py
## Not in use
from fastapi import APIRouter, HTTPException, Header, status
from typing import Optional
from pydantic import BaseModel
from services.chat_service import ChatService

router = APIRouter()

# Request/Response Models
class CreateChatRequest(BaseModel):
    message: str

class AddMessageRequest(BaseModel):
    role: str
    content: str
    references: Optional[list] = None

class UpdateChatTitleRequest(BaseModel):
    title: str

# ============================================================================
# /api/chats - Get all chats & Create new chat & Delete all chats
# ============================================================================

@router.get("")
async def get_user_chats(x_user_id: Optional[str] = Header(None)):
    """Get all chats for a user"""
    try:
        user_id = x_user_id or "default-user"
        print(f"Fetching chats for userId: {user_id}")
        
        chats = await ChatService.get_user_chats(user_id)
        return {"chats": chats}
    
    except Exception as e:
        print(f"Error fetching chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chats"
        )

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_chat(
    request: CreateChatRequest,
    x_user_id: Optional[str] = Header(None)
):
    """Create a new chat"""
    try:
        user_id = x_user_id or "default-user"
        
        if not request.message or not isinstance(request.message, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message is required"
            )
        
        chat = await ChatService.create_chat(user_id, request.message)
        return {"chat": chat}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat"
        )

@router.delete("")
async def delete_all_chats(x_user_id: Optional[str] = Header(None)):
    """Delete all chats for a user"""
    try:
        user_id = x_user_id or "default-user"
        
        deleted_count = await ChatService.delete_all_chats(user_id)
        return {
            "success": True,
            "deletedCount": deleted_count
        }
    
    except Exception as e:
        print(f"Error deleting all chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete all chats"
        )

# ============================================================================
# /api/chats/{chat_id} - Get/Delete/Update specific chat
# ============================================================================

@router.get("/{chat_id}")
async def get_chat_by_id(
    chat_id: str,
    x_user_id: Optional[str] = Header(None)
):
    """Get a specific chat by ID"""
    try:
        user_id = x_user_id or "default-user"
        print(f"Fetching chat for chatId: {chat_id}")
        
        chat = await ChatService.get_chat_by_id(chat_id, user_id)
        
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        return {"chat": chat}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch chat"
        )

@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    x_user_id: Optional[str] = Header(None)
):
    """Delete a specific chat"""
    try:
        user_id = x_user_id or "default-user"
        
        deleted = await ChatService.delete_chat(chat_id, user_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        return {"success": True}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat"
        )

@router.patch("/{chat_id}")
async def update_chat_title(
    chat_id: str,
    request: UpdateChatTitleRequest,
    x_user_id: Optional[str] = Header(None)
):
    """Update chat title"""
    try:
        user_id = x_user_id or "default-user"
        
        if not request.title or not isinstance(request.title, str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title is required"
            )
        
        updated = await ChatService.update_chat_title(
            chat_id, user_id, request.title
        )
        
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        
        return {"success": True}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update chat"
        )

# ============================================================================
# /api/chats/{chat_id}/messages - Get/Add messages
# ============================================================================

@router.get("/{chat_id}/messages")
async def get_chat_messages(chat_id: str):
    """Get all messages for a chat"""
    try:
        print(f"Fetching messages for chatId: {chat_id}")
        
        messages = await ChatService.get_chat_messages(chat_id)
        return {"messages": messages}
    
    except Exception as e:
        print(f"Error fetching messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch messages"
        )

@router.post("/{chat_id}/messages", status_code=status.HTTP_201_CREATED)
async def add_message(
    chat_id: str,
    request: AddMessageRequest
):
    """Add a message to a chat"""
    try:
        if not request.role or not request.content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role or content"
            )
        
        if request.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role must be 'user' or 'assistant'"
            )
        
        message = await ChatService.add_message(
            chat_id,
            request.role,
            request.content,
            request.references
        )
        
        return {"message": message}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error adding message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add message"
        )