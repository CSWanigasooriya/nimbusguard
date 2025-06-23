import os
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor

# Initialize OpenTelemetry tracing
def setup_tracing():
    # Configure the tracer provider with service information
    resource = Resource(attributes={
        SERVICE_NAME: "nimbusguard-consumer",
        "service.version": "1.0.0",
        "service.namespace": "nimbusguard"
    })
    
    # Set up trace provider
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    
    # Configure OTLP exporter (sends traces to Alloy)
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://alloy:4318/v1/traces"),
        headers={}
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    return tracer

# Set up tracing before creating the FastAPI app
tracer = setup_tracing()

# Create FastAPI instance
app = FastAPI(
    title="NimbusGuard Consumer",
    description="A consumer service for NimbusGuard",
    version="1.0.0"
)

# Initialize Prometheus metrics instrumentation
Instrumentator().instrument(app).expose(app)

# Auto-instrument FastAPI for tracing (captures all HTTP requests automatically)
FastAPIInstrumentor.instrument_app(app)

# Auto-instrument requests library for outbound HTTP calls
RequestsInstrumentor().instrument()

# Auto-instrument urllib3 for HTTP client calls
URLLib3Instrumentor().instrument()

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