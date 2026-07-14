"""Tracing setup (Phase 8).

- ``setup_langsmith``: enables LangChain/LangGraph -> LangSmith tracing, but ONLY when
  ``LANGCHAIN_API_KEY`` is set. With no key it is a no-op, so the app runs cleanly
  until you add the key (then every node + LLM call is traced automatically).
- ``setup_otel``: instruments the FastAPI app with OpenTelemetry request spans; exports
  them via OTLP only if ``OTEL_EXPORTER_OTLP_ENDPOINT`` is configured.
"""

import logging
import os

from app.config import settings

logger = logging.getLogger("studyhelper.observability")


def setup_langsmith() -> bool:
    """Enable LangSmith tracing if an API key is configured. Returns whether enabled."""
    if not settings.LANGCHAIN_API_KEY:
        logger.info("LangSmith tracing OFF (set LANGCHAIN_API_KEY in .env to enable).")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    logger.info("LangSmith tracing ON (project=%s).", settings.LANGCHAIN_PROJECT)
    return True


def setup_otel(app) -> bool:
    """Instrument FastAPI with OpenTelemetry. Export via OTLP only if configured."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except Exception as e:  # instrumentation package missing
        logger.warning("OpenTelemetry unavailable: %s", e)
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": "studyhelper"}))
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT))
        )
        logger.info("OTel exporting spans -> %s", settings.OTEL_EXPORTER_OTLP_ENDPOINT)
    else:
        logger.info("OTel instrumented (no exporter endpoint set; spans not exported).")

    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    return True
