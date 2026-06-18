import streamlit as st
from groq import Groq
from traceloop.sdk import Traceloop

# Streamlit Cloud stores secrets in st.secrets; fall back to env vars for local dev.
def _secret(key: str, default: str = "") -> str:
    import os
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

# Initialize Traceloop once per process — auto-instruments the Groq SDK
# and exports OTel Gen AI semantic convention spans to the configured OTLP endpoint.
if "traceloop_initialized" not in st.session_state:
    Traceloop.init(
        app_name="dash0-llm-demo",
        api_endpoint=_secret("DASH0_OTLP_ENDPOINT"),
        headers={
            "Authorization": f"Bearer {_secret('DASH0_AUTH_TOKEN')}",
            "Dash0-Dataset": _secret("DASH0_DATASET", "default"),
        },
        disable_batch=False,
    )
    st.session_state.traceloop_initialized = True

client = Groq(api_key=_secret("GROQ_API_KEY"))

# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dash0 LLM Tracing Demo", page_icon="🔭", layout="centered")

st.title("🔭 Dash0 LLM Tracing Demo")
st.caption(
    "Every message is traced with [OpenTelemetry Gen AI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) "
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

# Chat history
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

        # Streaming call — Traceloop intercepts and emits OTel spans automatically
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
