from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI(
    title="NimbusGuard Consumer",
    description="A consumer service for NimbusGuard",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message"""
    return {"message": "Welcome to NimbusGuard Consumer!"}

@app.get("/health")
async def health():
    """Liveness probe endpoint"""
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    """Readiness probe endpoint"""
    return {"status": "ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 