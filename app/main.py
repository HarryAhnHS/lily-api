from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import students, objectives, sessions, goals, subject_areas, iep_upload, transcript, weekly_summary
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    redirect_slashes=False,
    title="Mirae API",
    description="API for Mirae application",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "IEP Upload",
            "description": "Operations with IEP PDF uploads",
        },
    ],
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=[
    #     os.getenv("FRONTEND_URL"),
    # ],
    allow_origins=[os.getenv("FRONTEND_URL")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(students.router, prefix="/students")
app.include_router(subject_areas.router, prefix="/subject-areas")
app.include_router(goals.router, prefix="/goals")
app.include_router(objectives.router, prefix="/objectives")
app.include_router(sessions.router, prefix="/sessions")
app.include_router(iep_upload.router, prefix="/iep-upload")
app.include_router(transcript.router, prefix="/transcript")
app.include_router(weekly_summary.router, prefix="/weekly-summary")