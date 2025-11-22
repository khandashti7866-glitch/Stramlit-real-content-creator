# app.py
"""
AI SOCIAL MEDIA CONTENT CREATOR (Luxury Black & Gold UI)
--------------------------------------------------------
Run:
    streamlit run app.py
"""

import streamlit as st
from datetime import datetime
import time
import random
import sys

# ========================================================
#  CUSTOM LUXURY BLACK + GOLD THEME + BACKGROUND
# ========================================================
BACKGROUND_IMAGE_URL = "https://image2url.com/images/1763826608583-b523e17b-8a25-47e0-8ca0-c0d7245abed7.jpg"   # CHANGE THIS

luxury_css = f"""
<style>

html, body, [class*="css"]  {{
    background: url("{BACKGROUND_IMAGE_URL}") !important;
    background-size: cover !important;
    background-attachment: fixed !important;
}}

[data-testid="stAppViewContainer"] {{
    background: rgba(0, 0, 0, 0.70) !important;
    backdrop-filter: blur(6px);
}}

[data-testid="stSidebar"] {{
    background: rgba(0, 0, 0, 0.85) !important;
    backdrop-filter: blur(8px);
    border-right: 2px solid gold;
}}

h1, h2, h3, h4, h5, h6, label, p, span {{
    color: gold !important;
}}

textarea, input {{
    background: #111 !important;
    color: gold !important;
    border: 1px solid gold !important;
}}

.stButton>button {{
    background: linear-gradient(135deg, #e6b800, #b38600) !important;
    color: black !important;
    border-radius: 10px !important;
    border: 1px solid gold !important;
    font-weight: bold !important;
}}

.chat-bubble-user {{
    background: rgba(255, 255, 255, 0.12);
    padding: 10px;
    border-left: 4px solid #ffcc00;
    border-radius: 6px;
    color: white;
}}

.chat-bubble-ai {{
    background: rgba(255, 215, 0, 0.18);
    padding: 10px;
    border-left: 4px solid gold;
    border-radius: 6px;
    color: white;
}}

</style>
"""
st.markdown(luxury_css, unsafe_allow_html=True)

# ========================================================
# MODEL SETTINGS (same as your code)
# ========================================================
MODEL_NAME = "sshleifer/tiny-gpt2"
INCLUDE_PROMPT_IN_GENERATION = True
CONTEXT_MESSAGES = 6
FALLBACK_SEED = 42

MODEL_AVAILABLE = False
generator = None
tokenizer = None
device = -1

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, set_seed
    import torch
    device = 0 if torch.cuda.is_available() else -1

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        MODEL_AVAILABLE = True
    except Exception as load_exc:
        print("Model load failed:", load_exc, file=sys.stderr)
        MODEL_AVAILABLE = False

except Exception as e:
    print("Transformers import failed:", e, file=sys.stderr)
    MODEL_AVAILABLE = False

# ========================================================
# FALLBACK GENERATOR (same as your code)
# ========================================================
random.seed(FALLBACK_SEED)

def fallback_generate(prompt: str, max_tokens: int = 200, temperature: float = 0.7, style: str = "Friendly") -> str:
    templates = {
        "Friendly": [
            "Here's something helpful:\n\n{body}",
            "Okay! Try this:\n\n{body}"
        ],
        "Professional": [
            "Professional answer:\n\n{body}",
            "Analysis:\n\n{body}"
        ],
        "Short": [
            "{body}",
            "Quick answer: {body}"
        ],
        "Motivational": [
            "Believe in yourself:\n\n{body}",
            "Motivational answer:\n\n{body}"
        ],
    }

    try:
        last_line = prompt.strip().splitlines()[-1]
    except:
        last_line = ""

    body = f"I understand \"{last_line}\". Here is a clear, helpful suggestion."

    template = random.choice(templates.get(style, templates["Friendly"]))
    reply = template.format(body=body)

    if "code" in prompt.lower():
        reply += "\n\nExample:\n```\nprint('Hello!')\n```"

    reply_words = reply.split()
    if len(reply_words) > max_tokens:
        reply = " ".join(reply_words[:max_tokens]) + " ..."

    return reply

# ========================================================
# PROMPT BUILDER + GENERATION
# ========================================================
def compose_prompt(history, system_instruction, style):
    parts = []
    if system_instruction:
        parts.append(f"System: {system_instruction}")

    recent = history[-CONTEXT_MESSAGES:]
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{role}: {m['text']}")

    parts.append("Assistant:")
    return "\n".join(parts)

def generate_response(history, system_instruction, user_message, max_tokens, temperature, style):
    prompt = compose_prompt(history + [{"role": "user", "text": user_message}], system_instruction, style)

    if MODEL_AVAILABLE and generator:
        try:
            set_seed(int(time.time()))
            input_text = prompt if INCLUDE_PROMPT_IN_GENERATION else user_message

            outputs = generator(
                input_text,
                max_length=max_tokens + 50,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1,
            )

            raw = outputs[0]["generated_text"]
            reply = raw[len(input_text):].strip() if raw.startswith(input_text) else raw.strip()

            if not reply:
                reply = fallback_generate(prompt)

            return reply

        except Exception as e:
            print("Model error:", e)
            return fallback_generate(prompt)

    return fallback_generate(prompt)

# ========================================================
# STREAMlit UI
# ========================================================
st.set_page_config(page_title="AI SOCIAL MEDIA CONTENT CREATOR", layout="wide")
st.title("✨ LUXURY AI SOCIAL MEDIA CONTENT CREATOR (NO API KEY)")

# ---------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙ Settings")
    system_instruction = st.text_area("System Prompt:", "You are a helpful assistant.", height=80)
    style = st.selectbox("Assistant Style", ["Friendly", "Professional", "Short", "Motivational"])
    max_tokens = st.slider("Max Tokens", 50, 1024, 200, 50)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    st.write("Model:", "**Loaded**" if MODEL_AVAILABLE else "**Fallback**")

    if st.button("Clear Chat"):
        st.session_state["chat_history"] = [
            {"role": "system", "text": system_instruction, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ]
        st.rerun()

# -------------- Initialize Chat ----------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "system", "text": "You are a helpful assistant.",
         "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    ]

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat")

    # Chat Display
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>🧑 You ({msg['time']}):<br>{msg['text']}</div><br>", unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            st.markdown(f"<div class='chat-bubble-ai'>🤖 AI ({msg['time']}):<br>{msg['text']}</div><br>", unsafe_allow_html=True)
        else:
            st.warning(f"System: {msg['text']}")

    st.markdown("---")

    # Chat Input
    with st.form("chat_form", clear_on_submit=False):
        user_input = st.text_area("Your Message", height=120)
        send = st.form_submit_button("Send")

        if send and user_input.strip():
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append({"role": "user", "text": user_input, "time": now})

            reply = generate_response(
                st.session_state.chat_history,
                system_instruction,
                user_input,
                max_tokens,
                temperature,
                style,
            )

            now2 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append({"role": "assistant", "text": reply, "time": now2})

            st.rerun()

with col2:
    st.subheader("⚡ Quick Prompts")
    if st.button("Explain Code"):
        st.session_state.chat_history.append({"role": "user", "text": "Explain this Python code.", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        st.rerun()
    if st.button("Summarize Text"):
        st.session_state.chat_history.append({"role": "user", "text": "Summarize this text.", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        st.rerun()
    if st.button("Generate Ideas"):
        st.session_state.chat_history.append({"role": "user", "text": "Give 5 content ideas.", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        st.rerun()

st.markdown("---")
st.caption("✨ Luxury Black & Gold Offline AI — No API Key Required.")

