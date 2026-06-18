import logging
import streamlit as st
from groq import Groq
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.trace import SpanKind, StatusCode


def _secret(key: str, default: str = "") -> str:
    import os
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)


def _setup_otel() -> None:
    endpoint = _secret("DASH0_OTLP_ENDPOINT").rstrip("/")
    headers = {
        "Authorization": f"Bearer {_secret('DASH0_AUTH_TOKEN')}",
        "Dash0-Dataset": _secret("DASH0_DATASET", "default"),
    }
    resource = Resource.create({
        "service.name": "dash0-llm-demo",
        "service.version": "1.0.0",
    })

    # ── Traces ──────────────────────────────────────────────────────────────
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", headers=headers)
    ))
    trace.set_tracer_provider(tp)

    # ── Logs (correlated to traces via trace_id / span_id) ──────────────────
    lp = LoggerProvider(resource=resource)
    lp.add_log_record_processor(BatchLogRecordProcessor(
        OTLPLogExporter(endpoint=f"{endpoint}/v1/logs", headers=headers)
    ))
    set_logger_provider(lp)
    root = logging.getLogger()
    root.addHandler(LoggingHandler(logger_provider=lp))
    root.setLevel(logging.INFO)


# Initialize once per process — safe across Streamlit reruns
if not isinstance(trace.get_tracer_provider(), TracerProvider):
    _setup_otel()

tracer = trace.get_tracer("dash0-llm-demo", "1.0.0")
logger = logging.getLogger("dash0.llm.demo")
client = Groq(api_key=_secret("GROQ_API_KEY"))

# ── UI ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dash0 LLM Tracing Demo", page_icon="🔭", layout="centered")

st.title("🔭 Dash0 LLM Tracing Demo")
st.caption(
    "Every message is traced with [OpenTelemetry Gen AI semantic conventions]"
    "(https://opentelemetry.io/docs/specs/semconv/gen-ai/) "
    "and sent live to [Dash0](https://www.dash0.com)."
)

MODEL_OPTIONS = {
    "Llama 3.1 8B Instant (fastest)": "llama-3.1-8b-instant",
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Gemma 2 9B": "gemma2-9b-it",
}
selected_label = st.sidebar.selectbox("Model", list(MODEL_OPTIONS.keys()))
model_name = MODEL_OPTIONS[selected_label]

system_prompt = st.sidebar.text_area(
    "System prompt",
    value="You are a helpful assistant. Be concise.",
    height=120,
)

max_tokens = st.sidebar.slider("Max tokens", 64, 2048, 512, 64)

st.sidebar.divider()
st.sidebar.markdown("**Traces appear in your Dash0 dashboard in real time.**")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with tracer.start_as_current_span(
            f"chat {model_name}", kind=SpanKind.CLIENT
        ) as span:
            # ── Request attributes ───────────────────────────────────────────
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.system", "groq")
            span.set_attribute("gen_ai.request.model", model_name)
            span.set_attribute("gen_ai.request.max_tokens", max_tokens)
            span.set_attribute("server.address", "api.groq.com")
            span.set_attribute("server.port", 443)

            # ── Input message events ─────────────────────────────────────────
            span.add_event("gen_ai.system.message", {"content": system_prompt})
            for msg in st.session_state.messages[:-1]:
                event = "gen_ai.user.message" if msg["role"] == "user" else "gen_ai.assistant.message"
                span.add_event(event, {"content": msg["content"]})
            span.add_event("gen_ai.user.message", {"content": prompt})

            try:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "system", "content": system_prompt}]
                    + st.session_state.messages,
                    max_tokens=max_tokens,
                    stream=True,
                )

                input_tokens = output_tokens = 0
                finish_reason = "stop"
                response_id = response_model = None

                for chunk in stream:
                    if chunk.id and not response_id:
                        response_id = chunk.id
                        response_model = chunk.model
                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens or 0
                        output_tokens = chunk.usage.completion_tokens or 0
                    if chunk.choices and chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")

                # ── Response attributes ──────────────────────────────────────
                if response_id:
                    span.set_attribute("gen_ai.response.id", response_id)
                span.set_attribute("gen_ai.response.model", response_model or model_name)
                span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

                # ── Choice event ─────────────────────────────────────────────
                span.add_event("gen_ai.choice", {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message.role": "assistant",
                    "message.content": full_response,
                })

                # ── Correlated log ───────────────────────────────────────────
                logger.info(
                    "chat completion",
                    extra={
                        "gen_ai.system": "groq",
                        "gen_ai.request.model": model_name,
                        "gen_ai.response.model": response_model or model_name,
                        "gen_ai.usage.input_tokens": input_tokens,
                        "gen_ai.usage.output_tokens": output_tokens,
                        "gen_ai.response.finish_reason": finish_reason,
                    },
                )

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
