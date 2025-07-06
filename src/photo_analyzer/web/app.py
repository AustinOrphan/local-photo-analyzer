"""FastAPI web application for photo analyzer."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import uvicorn

from photo_analyzer.core.config import get_config
from photo_analyzer.core.logger import get_logger, audit_log
from photo_analyzer.database.session import get_db_dependency
from photo_analyzer.pipeline.analyzer import PhotoAnalyzer
from photo_analyzer.pipeline.processor import PhotoProcessor
from photo_analyzer.pipeline.organizer import PhotoOrganizer
from photo_analyzer.models.photo import Photo
from photo_analyzer.web.schemas import (
    PhotoResponse, AnalysisRequest, AnalysisResponse,
    OrganizationRequest, OrganizationResponse, BatchRequest
)
from photo_analyzer.web.advanced_routes import router as advanced_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Photo Analyzer web application")
    
    # Initialize components
    app.state.config = get_config()
    app.state.analyzer = PhotoAnalyzer()
    app.state.processor = PhotoProcessor()
    app.state.organizer = PhotoOrganizer()
    
    # Check Ollama connection
    if await app.state.analyzer.llm_client.check_connection():
        logger.info("Successfully connected to Ollama")
    else:
        logger.warning("Could not connect to Ollama - some features may be unavailable")
    
    audit_log("APPLICATION_START")
    
    yield
    
    logger.info("Shutting down Photo Analyzer web application")
    audit_log("APPLICATION_STOP")


app = FastAPI(
    title="Photo Analyzer",
    description="Secure local LLM-based photo analyzer with intelligent organization",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include advanced routes
app.include_router(advanced_router)


# Dependencies
def get_analyzer() -> PhotoAnalyzer:
    """Get photo analyzer instance."""
    return app.state.analyzer

def get_processor() -> PhotoProcessor:
    """Get photo processor instance."""
    return app.state.processor

def get_organizer() -> PhotoOrganizer:
    """Get photo organizer instance."""
    return app.state.organizer


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}

@app.get("/health/ollama")
async def ollama_health(analyzer: PhotoAnalyzer = Depends(get_analyzer)):
    """Check Ollama connection health."""
    connected = await analyzer.llm_client.check_connection()
    return {"ollama_connected": connected}


# Photo management endpoints
@app.get("/api/photos", response_model=List[PhotoResponse])
async def list_photos(
    limit: int = 50,
    offset: int = 0,
    tag: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    session: AsyncSession = Depends(get_db_dependency)
):
    """List photos with optional filtering."""
    try:
        # Basic query - in a real implementation, you'd add filtering
        # This is a simplified version for the initial implementation
        photos = []  # Placeholder - implement actual database query
        
        return [PhotoResponse.from_orm(photo) for photo in photos]
    
    except Exception as e:
        logger.error(f"Error listing photos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/photos/{photo_id}", response_model=PhotoResponse)
async def get_photo(
    photo_id: str,
    session: AsyncSession = Depends(get_db_dependency)
):
    """Get a specific photo by ID."""
    try:
        # Placeholder - implement actual database query
        photo = None  # Query photo by ID
        
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        return PhotoResponse.from_orm(photo)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting photo {photo_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/photos/{photo_id}/thumbnail")
async def get_photo_thumbnail(photo_id: str):
    """Get photo thumbnail."""
    try:
        # Placeholder - implement thumbnail generation/retrieval
        # For now, return a placeholder response
        raise HTTPException(status_code=501, detail="Thumbnail generation not implemented yet")
    
    except Exception as e:
        logger.error(f"Error getting thumbnail for photo {photo_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Photo upload endpoint
@app.post("/api/photos/upload")
async def upload_photo(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_analyze: bool = True,
    auto_organize: bool = False,
    processor: PhotoProcessor = Depends(get_processor)
):
    """Upload a new photo."""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file temporarily
        config = get_config()
        temp_dir = config.cache_dir / "uploads"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = temp_dir / file.filename
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        audit_log("PHOTO_UPLOAD", filename=file.filename, size=len(content))
        
        # Process the photo
        photo_info = await processor.process_photo(temp_file)
        
        # Schedule background analysis if requested
        if auto_analyze:
            background_tasks.add_task(
                analyze_photo_background,
                photo_info['id'],
                auto_organize
            )
        
        return {
            "message": "Photo uploaded successfully",
            "photo_id": photo_info['id'],
            "filename": file.filename,
            "auto_analyze": auto_analyze
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analysis endpoints
@app.post("/api/photos/{photo_id}/analyze", response_model=AnalysisResponse)
async def analyze_photo(
    photo_id: str,
    request: AnalysisRequest,
    analyzer: PhotoAnalyzer = Depends(get_analyzer),
    session: AsyncSession = Depends(get_db_dependency)
):
    """Analyze a specific photo."""
    try:
        # Get photo from database
        # Placeholder - implement actual database query
        photo = None  # Query photo by ID
        
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        # Perform analysis
        analysis_result = await analyzer.analyze_photo(
            photo.current_path,
            model=request.model,
            include_tags=request.include_tags,
            include_description=request.include_description,
            include_filename_suggestion=request.include_filename_suggestion
        )
        
        audit_log("PHOTO_ANALYSIS", photo_id=photo_id, model=request.model)
        
        return AnalysisResponse(**analysis_result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing photo {photo_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/photos/analyze/batch", response_model=List[AnalysisResponse])
async def analyze_photos_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    analyzer: PhotoAnalyzer = Depends(get_analyzer)
):
    """Analyze multiple photos in batch."""
    try:
        if len(request.photo_ids) > 100:
            raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
        
        # Schedule background batch analysis
        background_tasks.add_task(
            analyze_batch_background,
            request.photo_ids,
            request.model
        )
        
        audit_log("BATCH_ANALYSIS_STARTED", count=len(request.photo_ids))
        
        return JSONResponse({
            "message": f"Batch analysis started for {len(request.photo_ids)} photos",
            "photo_ids": request.photo_ids
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Organization endpoints
@app.post("/api/photos/{photo_id}/organize", response_model=OrganizationResponse)
async def organize_photo(
    photo_id: str,
    request: OrganizationRequest,
    organizer: PhotoOrganizer = Depends(get_organizer),
    session: AsyncSession = Depends(get_db_dependency)
):
    """Organize a specific photo."""
    try:
        # Get photo from database
        # Placeholder - implement actual database query
        photo = None  # Query photo by ID
        
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        # Organize the photo
        result = await organizer.organize_photo(
            photo,
            target_structure=request.target_structure,
            create_symlinks=request.create_symlinks,
            backup_original=request.backup_original
        )
        
        audit_log("PHOTO_ORGANIZATION", photo_id=photo_id, structure=request.target_structure)
        
        return OrganizationResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error organizing photo {photo_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Search endpoints
@app.get("/api/search")
async def search_photos(
    q: str,
    limit: int = 20,
    session: AsyncSession = Depends(get_db_dependency)
):
    """Search photos by content, tags, or filename."""
    try:
        if len(q.strip()) < 2:
            raise HTTPException(status_code=400, detail="Search query too short")
        
        # Placeholder - implement actual search functionality
        results = []  # Implement search logic
        
        audit_log("PHOTO_SEARCH", query=q, results_count=len(results))
        
        return {
            "query": q,
            "results": results,
            "total": len(results)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching photos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def analyze_photo_background(photo_id: str, auto_organize: bool = False):
    """Background task for photo analysis."""
    try:
        analyzer = app.state.analyzer
        organizer = app.state.organizer if auto_organize else None
        
        # Implement background analysis logic
        logger.info(f"Starting background analysis for photo {photo_id}")
        
        # Placeholder - implement actual analysis
        # analysis_result = await analyzer.analyze_photo(photo_path)
        
        if auto_organize and organizer:
            # Placeholder - implement organization
            logger.info(f"Auto-organizing photo {photo_id}")
        
        audit_log("BACKGROUND_ANALYSIS_COMPLETE", photo_id=photo_id)
        
    except Exception as e:
        logger.error(f"Background analysis failed for photo {photo_id}: {e}")
        audit_log("BACKGROUND_ANALYSIS_ERROR", photo_id=photo_id, error=str(e))


async def analyze_batch_background(photo_ids: List[str], model: Optional[str] = None):
    """Background task for batch analysis."""
    try:
        analyzer = app.state.analyzer
        
        logger.info(f"Starting batch analysis for {len(photo_ids)} photos")
        
        # Placeholder - implement actual batch analysis
        # results = await analyzer.analyze_batch(photo_paths, model=model)
        
        audit_log("BATCH_ANALYSIS_COMPLETE", count=len(photo_ids))
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        audit_log("BATCH_ANALYSIS_ERROR", count=len(photo_ids), error=str(e))


# Static files for frontend (when built)
try:
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        @app.get("/")
        async def serve_frontend():
            """Serve the React frontend."""
            index_file = static_dir / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return {"message": "Photo Analyzer API - Frontend not built"}
    else:
        @app.get("/")
        async def api_info():
            """API information endpoint."""
            return {
                "name": "Photo Analyzer API",
                "version": "0.1.0",
                "docs": "/docs",
                "health": "/health"
            }
except Exception as e:
    logger.warning(f"Could not set up static file serving: {e}")


def run_dev_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Run the development server."""
    uvicorn.run(
        "photo_analyzer.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_dev_server()