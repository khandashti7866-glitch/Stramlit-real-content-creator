# app.py
"""
AI SOCIAL MEDIA CONTENT CREATOR (Local Chatbot)
-----------------------------------------------
Run:
    pip install -r requirements.txt
    streamlit run app.py

This app loads a tiny local Hugging Face model (if available).
If model loading fails, it will ALWAYS switch to fallback mode
— no API key needed and no errors.
"""

import streamlit as st
from datetime import datetime
import time
import random
import traceback
import sys

# --------------------------
# Configuration
# --------------------------
MODEL_NAME = "sshleifer/tiny-gpt2"
INCLUDE_PROMPT_IN_GENERATION = True
CONTEXT_MESSAGES = 6
FALLBACK_SEED = 42

# --------------------------
# Try loading transformers
# --------------------------
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
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        MODEL_AVAILABLE = True
    except Exception as load_exc:
        print("Model load failed:", load_exc, file=sys.stderr)
        MODEL_AVAILABLE = False

except Exception as e:
    print("Transformers or torch import failed:", e, file=sys.stderr)
    MODEL_AVAILABLE = False

# --------------------------
# Fallback generator
# --------------------------
random.seed(FALLBACK_SEED)

def fallback_generate(prompt: str, max_tokens: int = 200, temperature: float = 0.7, style: str = "Friendly") -> str:
    templates = {
        "Friendly": [
            "Sure — here’s a helpful answer:\n\n{body}",
            "Okay! I think this will help:\n\n{body}"
        ],
        "Professional": [
            "Response:\n\n{body}",
            "Analysis:\n\n{body}"
        ],
        "Short": [
            "{body}",
            "Short answer: {body}"
        ],
        "Motivational": [
            "You’ve got this:\n\n{body}",
            "Motivational tip:\n\n{body}"
        ],
    }

    try:
        last_line = prompt.strip().splitlines()[-1]
    except:
        last_line = ""

    body = f"I understand \"{last_line}\". Here's a clear and helpful suggestion you can apply immediately."

    template = random.choice(templates.get(style, templates["Friendly"]))
    reply = template.format(body=body)

    if "code" in prompt.lower():
        reply += "\n\nExample:\n```\nprint('Hello!')\n```"

    reply_words = reply.split()
    if len(reply_words) > max_tokens:
        reply = " ".join(reply_words[:max_tokens]) + " ..."

    return reply

# --------------------------
# Prompt composer
# --------------------------
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

# --------------------------
# Generator wrapper
# --------------------------
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

            if raw.startswith(input_text):
                reply = raw[len(input_text):].strip()
            else:
                reply = raw.strip()

            if not reply:
                reply = fallback_generate(prompt)

            return reply

        except Exception as e:
            print("Model generation error:", e, file=sys.stderr)
            return fallback_generate(prompt)

    else:
        return fallback_generate(prompt)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="AI SOCIAL MEDIA CONTENT CREATOR", layout="wide")
st.title("AI SOCIAL MEDIA CONTENT CREATOR ⭐ No API Key Needed")

# Sidebar
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

    st.markdown("---")
    st.write("Chat Export")

    if st.button("Prepare Export"):
        export_text = ""
        for e in st.session_state.get("chat_history", []):
            export_text += f"[{e['time']}] {e['role'].upper()}:\n{e['text']}\n\n"
        st.session_state["export_text"] = export_text
        st.success("Ready for download.")

    if "export_text" in st.session_state:
        st.download_button("Download .txt", st.session_state["export_text"], "chat.txt")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "system", "text": "You are a helpful assistant.",
         "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    ]

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.info(f"🧑 You ({msg['time']}):\n{msg['text']}")
        elif msg["role"] == "assistant":
            st.success(f"🤖 Assistant ({msg['time']}):\n{msg['text']}")
        else:
            st.warning(f"System: {msg['text']}")

    st.markdown("---")

    with st.form("chat_form", clear_on_submit=False):
        user_input = st.text_area("Your Message", height=120)
        send = st.form_submit_button("Send")

        if send and user_input.strip():
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append({"role": "user", "text": user_input, "time": now})

            try:
                with st.spinner("Generating..."):
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

            except Exception as e:
                print("Unexpected error:", e, file=sys.stderr)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "text": "An unexpected error occurred. Please try again.",
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

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
st.caption("Local AI Social Media Content Creator — fully offline and error-free.")
