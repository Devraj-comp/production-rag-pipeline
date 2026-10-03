from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.services.rag_service import rag_service
import os

router = APIRouter()

@router.post("/ingest")
async def ingest_file(file_path: str, background_tasks: BackgroundTasks):
    """Trigger document ingestion in the background."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found at provided path")

    background_tasks.add_task(rag_service.ingest_document, file_path)
    return {"message": "Ingestion started in background", "file": file_path}

@router.get("/query")
async def query_rag(q: str):
    """Ask a question based on the ingested local documents."""
    try:
        result = await rag_service.query(q)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
