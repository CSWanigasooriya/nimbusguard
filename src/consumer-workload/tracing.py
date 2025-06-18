import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

def setup_tracing():
    """
    Configures the OpenTelemetry SDK to send traces to an OTLP collector.
    """
    # Get the OTLP endpoint from an environment variable.
    # This is the address of your Grafana Alloy service.
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not otlp_endpoint:
        logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT not set. Tracing will be disabled.")
        return

    # Set a "Resource" to identify your service. This adds a 'service.name'
    # attribute to all traces, which is crucial for filtering in Grafana/Tempo.
    service_name = os.getenv("OTEL_SERVICE_NAME", "nimbusguard-consumer-workload")
    resource = Resource(attributes={
        "service.name": service_name
    })

    # Set up the tracer provider
    provider = TracerProvider(resource=resource)

    # Create an OTLP exporter
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)

    # Use a BatchSpanProcessor to send traces in the background
    processor = BatchSpanProcessor(exporter)

    # Add the processor to the provider
    provider.add_span_processor(processor)

    # Set the global tracer provider
    trace.set_tracer_provider(provider)
    logger.info(f"Tracing enabled. Sending traces to {otlp_endpoint} with service name '{service_name}'.")