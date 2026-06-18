import streamlit as st
import google.generativeai as genai
from traceloop.sdk import Traceloop

# Streamlit Cloud stores secrets in st.secrets; fall back to env vars for local dev.
def _secret(key: str, default: str = "") -> str:
    import os
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)

# Initialize Traceloop once per process — auto-instruments the google-generativeai SDK
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

genai.configure(api_key=_secret("GEMINI_API_KEY"))

# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dash0 LLM Tracing Demo", page_icon="🔭", layout="centered")

st.title("🔭 Dash0 LLM Tracing Demo")
st.caption(
    "Every message is traced with [OpenTelemetry Gen AI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) "
    "and sent live to [Dash0](https://www.dash0.com)."
)

MODEL_OPTIONS = {
    "Gemini 2.0 Flash Lite (cheapest)": "gemini-2.0-flash-lite",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Gemini 1.5 Flash-8B": "gemini-1.5-flash-8b",
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

# Chat history (Streamlit format: role = "user" | "assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert history to Google's format (role "assistant" → "model", all but last message)
    history = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]}
        for m in st.session_state.messages[:-1]
    ]

    gemini_model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
    )
    chat = gemini_model.start_chat(history=history)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Streaming call — Traceloop intercepts and emits OTel spans automatically
        for chunk in chat.send_message(prompt, stream=True):
            full_response += chunk.text
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
