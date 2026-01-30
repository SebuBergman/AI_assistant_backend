import os
import tiktoken

from datetime import datetime
from uuid import uuid4
from fastapi import APIRouter, HTTPException, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import ReursiveCharacterTextSplitter

from app.config import _TOKEN_ENCODER, UPLOAD_PATH
from app.db.S3_bucket import upload_to_s3
from app.db.database import insert_pdf_metadata
from app.db.vectorstore_manager import COLLECTION_NAME, MILVUS_CONNECTION, get_vectorstore, embeddings

router = APIRouter()

text_splitter = ReursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_TOKEN_ENCODER.encode(text))

@router.post("/upload", tags=["RAG"])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    try:
        # Save file temporarily
        file_path = os.path.join(UPLOAD_PATH, file.filename)

        content = await file.read()
        file_size = len(content)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Upload to S3
        bucket_name = os.getenv("S3_BUCKET_NAME")
        s3_key = f"fetch_pdfs/{file.filename}"
        s3_url = upload_to_s3(file_path, bucket_name, s3_key)
        
        # Load and process PDF with LangChain
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        upload_date = datetime.now().isoformat()
        file_id = str(uuid4())
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                "file_name": file.filename,
                "source": s3_url,
                "file_size": file_size,
                "upload_date": upload_date,
                "page": doc.metadata.get("page", 0) + 1,  # convert to 1-based
                "file_id": file_id,
                "chunk_tokens": count_tokens(doc.page_content),
            })
        
        # Split documents
        splits = text_splitter.split_documents(documents)

        total_chunk_tokens = 0

        # Add chunk-level metadata
        for i, doc in enumerate(splits):
            chunk_tokens = count_tokens(doc.page_content)
            total_chunk_tokens += chunk_tokens

            doc.metadata.update({
                "chunk_id": f"{file_id}_{i}",
                "chunk_index": i,
                "chunk_tokens": chunk_tokens,
            })
        
        # Get or initialize vectorstore
        vs = get_vectorstore()
        if vs is None:
            from db.vectorstore_manager import vectorstore
            vs = Milvus(
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME,
                connection_args=MILVUS_CONNECTION,
                consistency_level="Strong",
                auto_id=True,
            )
            vectorstore = vs

        # Add documents to vectorstore
        print(f"Adding {len(splits)} documents to vectorstore...")
        vs.add_documents(splits)
        
        # Store PDF metadata in Milvus
        insert_pdf_metadata(
            file_name=file.filename,
            file_path=s3_url,
            file_size=file_size,
            upload_date=upload_date,
            file_id=file_id,
            chunks=len(splits),
            total_chunk_tokens=total_chunk_tokens
        )
        
        # Clean up local file
        os.remove(file_path)
        
        return {
            "message": f"Successfully processed {file.filename}",
            "s3_url": s3_url,
            "chunks_created": len(splits),
            "file_size": file_size,
            "upload_date": upload_date,
            "collection": COLLECTION_NAME
        }
    
    except Exception as e:
        print(f"Error in upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))