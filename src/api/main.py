"""
FastAPI Backend for Raga Music Generation
Main API server
"""

import os
import sys
import uuid
import asyncio
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add project root and src directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from sekiro_ai.services.gemini_parser import get_parser, MusicParameters
from sekiro_ai.services.audio_synth import get_synthesizer


# --- Pydantic Models ---

class GenerationRequest(BaseModel):
    """Request model for music generation"""
    text: str = Field(..., description="User description or lyrics", min_length=1, max_length=2000)
    instrument: str = Field(default="sitar", description="Instrument to use")
    raga_override: Optional[str] = Field(default=None, description="Optional raga override")
    tempo_override: Optional[int] = Field(default=None, ge=40, le=200, description="Optional tempo override (BPM)")
    duration_override: Optional[int] = Field(default=None, ge=10, le=300, description="Optional duration override (seconds)")


class GenerationResponse(BaseModel):
    """Response model for music generation"""
    job_id: str
    status: str
    message: str
    parameters: Optional[dict] = None


class JobStatus(BaseModel):
    """Status of a generation job"""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    midi_url: Optional[str] = None
    audio_url: Optional[str] = None
    parameters: Optional[dict] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class RagaInfo(BaseModel):
    """Information about a raga"""
    name: str
    moods: List[str]
    time_of_day: str
    tempo_range: List[int]
    description: str


# --- Application Setup ---

app = FastAPI(
    title="Raga Music Generator API",
    description="Generate Indian classical music from text descriptions using AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
jobs = {}  # In-memory job storage (use Redis in production)
parser = None
synthesizer = None
model = None
tokenizer = None
vocabularies = None


# --- Startup/Shutdown ---

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global parser, synthesizer, model, tokenizer, vocabularies
    
    # Initialize parser (will use mock if no API key)
    # Check for OpenRouter key first, then Gemini for backwards compatibility
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMINI_API_KEY")
    parser = get_parser(api_key)
    print("✅ LLM Parser initialized")
    
    # Initialize synthesizer
    synthesizer = get_synthesizer("soundfonts")
    print(f"✅ Audio synthesizer initialized. Instruments: {synthesizer.available_instruments}")
    
    # Load model (if checkpoint exists)
    # Support both model.pt (exported) and best_model.pt (from training)
    checkpoint_path = BASE_DIR / "checkpoints/model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = BASE_DIR / "checkpoints/best_model.pt"
    
    print(f"DEBUG: Looking for model at {checkpoint_path}")
    if checkpoint_path.exists():
        try:
            import torch
            from sekiro_ai.models.tokenizer import MIDITokenizer
            from sekiro_ai.models.transformer_cvae import RagaCVAE
            from sekiro_ai.models.dataset import RagaDataset
            
            # Load vocabularies
            vocab_path = BASE_DIR / "checkpoints/vocabularies.json"
            if vocab_path.exists():
                vocabularies = RagaDataset.load_vocabularies(str(vocab_path))
                print("✅ Vocabularies loaded")
            
            # Load tokenizer
            tokenizer_path = BASE_DIR / "checkpoints/tokenizer.json"
            if tokenizer_path.exists():
                tokenizer = MIDITokenizer.load(str(tokenizer_path))
                print("✅ Tokenizer loaded")
            
            # Load model (use weights_only=False for custom classes)
            # Try loading leveraging legacy support if needed
            try:
                # First try importing the new model
                from sekiro_ai.models.transformer_cvae import RagaCVAE
                
                print("DEBUG: Attempting to load model with current architecture...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                model = RagaCVAE(checkpoint['config'])
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Model loaded (Current Architecture)")
                
            except (RuntimeError, ModuleNotFoundError, AttributeError) as e_new:
                print(f"⚠️ Standard loading failed ({e_new}), attempting legacy load...")
                
                # Fallback to Legacy Model
                try:
                    import types
                    import importlib.util
                    
                    # Dynamic import from backend/services/legacy_model.py
                    legacy_path = BASE_DIR / "backend" / "services" / "legacy_model.py"
                    spec = importlib.util.spec_from_file_location("legacy_model", legacy_path)
                    legacy_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(legacy_module)
                    LegacyRagaCVAE = legacy_module.LegacyRagaCVAE
                    LegacyCVAEConfig = legacy_module.LegacyCVAEConfig
                    
                    # SHIM: Inject mock module for unpickling old checkpoints
                    if "transformer_cvae" not in sys.modules:
                        mock_cvae = types.ModuleType("transformer_cvae")
                        mock_cvae.CVAEConfig = LegacyCVAEConfig
                        mock_cvae.RagaCVAE = LegacyRagaCVAE
                        sys.modules["transformer_cvae"] = mock_cvae
                    
                    # Load checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    
                    # Inspect config to create correct model instance
                    if 'config' in checkpoint:
                        # If config is an object, copy attributes; if dict, use as is
                        loaded_cfg = checkpoint['config']
                        if isinstance(loaded_cfg, dict):
                            config = LegacyCVAEConfig(**loaded_cfg)
                        else:
                            # Re-create config from object dict
                            config = LegacyCVAEConfig(**loaded_cfg.__dict__)
                    else:
                        config = LegacyCVAEConfig()
                        
                    model = LegacyRagaCVAE(config)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    print("✅ Model loaded (Legacy Architecture)")
                    
                except Exception as e_legacy:
                    print(f"❌ Legacy model loading also failed: {e_legacy}")
                    import traceback
                    traceback.print_exc()
                    model = None
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            model = None
    else:
        print("⚠️ No trained model found. Train the model first!")


# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "parser_available": parser is not None,
        "synthesizer_available": synthesizer is not None
    }


@app.get("/instruments")
async def get_instruments():
    """Get available instruments"""
    return {
        "instruments": synthesizer.available_instruments if synthesizer else ["sitar"]
    }


@app.get("/ragas")
async def get_ragas():
    """Get available ragas with information"""
    import json
    
    import json
    
    config_path = BASE_DIR / "src/sekiro_ai/config/raga_metadata.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            raga_data = json.load(f)
        
        ragas = []
        for name, info in raga_data.items():
            ragas.append(RagaInfo(
                name=name,
                moods=info.get("moods", []),
                time_of_day=info.get("time_of_day", "any"),
                tempo_range=info.get("tempo_range", [60, 120]),
                description=info.get("description", "")
            ))
        return {"ragas": ragas}
    
    return {"ragas": []}


@app.post("/generate", response_model=GenerationResponse)
async def generate_music(request: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate music from text description
    Returns a job ID to track progress
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    
    # Parse text with LLM
    params = parser.parse_text(request.text)
    
    # Apply overrides
    if request.raga_override:
        params.raga = request.raga_override
    if request.tempo_override:
        params.tempo = request.tempo_override
    if request.duration_override:
        params.duration = request.duration_override
    
    # Create job
    job_id = str(uuid.uuid4())
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        parameters={
            "mood": params.mood,
            "tempo": params.tempo,
            "duration": params.duration,
            "raga": params.raga,
            "instrument": request.instrument
        },
        created_at=datetime.now().isoformat()
    )
    
    # Start background generation
    background_tasks.add_task(
        generate_music_task,
        job_id,
        params,
        request.instrument
    )
    
    return GenerationResponse(
        job_id=job_id,
        status="pending",
        message="Music generation started",
        parameters=jobs[job_id].parameters
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a generation job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/download/midi/{job_id}")
async def download_midi(job_id: str):
    """Download generated MIDI file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    midi_path = Path(f"outputs/{job_id}.mid")
    if not midi_path.exists():
        raise HTTPException(status_code=404, detail="MIDI file not found")
    
    return FileResponse(
        str(midi_path),
        media_type="audio/midi",
        filename=f"raga_{job.parameters['raga']}_{job_id[:8]}.mid"
    )


@app.get("/download/audio/{job_id}")
async def download_audio(job_id: str):
    """Download generated audio file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    audio_path = Path(f"outputs/{job_id}.wav")
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        str(audio_path),
        media_type="audio/wav",
        filename=f"raga_{job.parameters['raga']}_{job_id[:8]}.wav"
    )


# --- Background Task ---

async def generate_music_task(job_id: str, params: MusicParameters, instrument: str):
    """Background task for music generation with robust fallback handling"""
    from sekiro_ai.services.music_generator import MusicGenerator
    
    try:
        jobs[job_id].status = "processing"
        jobs[job_id].progress = 0.1
        
        # Create output directory
        output_dir = BASE_DIR / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        jobs[job_id].progress = 0.3
        
        # Use MusicGenerator which handles fallback automatically
        generator = MusicGenerator(model, tokenizer, vocabularies)
        
        # Generate music tokens (with automatic fallback if model fails)
        tokens = generator.generate(
            mood=params.mood,
            raga=params.raga,
            tempo=params.tempo,
            duration=params.duration,
            temperature=1.25, # Increased slightly for more dynamics
            top_k=60          # Increased for more variety
        )
        
        jobs[job_id].progress = 0.6
        
        # Convert tokens to MIDI
        midi_path = str(output_dir / f"{job_id}.mid")
        generator.tokens_to_midi(tokens, midi_path)
        
        jobs[job_id].progress = 0.8
        jobs[job_id].midi_url = f"/download/midi/{job_id}"
        
        # Try to convert MIDI to audio (may fail if FluidSynth not installed)
        audio_path = str(output_dir / f"{job_id}.wav")
        try:
            result = synthesizer.midi_to_audio(midi_path, audio_path, instrument)
            if result and Path(audio_path).exists():
                jobs[job_id].audio_url = f"/download/audio/{job_id}"
        except Exception as audio_err:
            print(f"Audio conversion skipped: {audio_err}")
            # Continue without audio - MIDI is still available
        
        jobs[job_id].progress = 1.0
        jobs[job_id].status = "completed"
        jobs[job_id].completed_at = datetime.now().isoformat()
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)


# --- Main ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
