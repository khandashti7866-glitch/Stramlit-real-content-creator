# app.py
"""
Streamlit Local Chatbot (single-file)
-------------------------------------
Run:
    pip install -r requirements.txt
    streamlit run app.py

This app tries to load a small open-source Hugging Face model locally (default: 'sshleifer/tiny-gpt2').
If loading or inference is not possible, the app uses a deterministic fallback generator that
supports multi-turn context so the app never crashes and requires NO API key.

Notes:
- Change MODEL_NAME near the top to try a different local model.
- If you have limited RAM or no internet, the fallback mode will be used automatically.
"""

import streamlit as st
from datetime import datetime
import time
import random
import traceback
import sys

# --------------------------
# Configuration - tweak here
# --------------------------
MODEL_NAME = "sshleifer/tiny-gpt2"  # change to a different HF model if you want (small models recommended)
INCLUDE_PROMPT_IN_GENERATION = True  # if True, we feed the composed prompt to the model
CONTEXT_MESSAGES = 6  # how many past messages to include in the prompt/context
FALLBACK_SEED = 42

# --------------------------
# Try to import transformers
# --------------------------
MODEL_AVAILABLE = False
generator = None
tokenizer = None
device = -1  # pipeline device (-1 for CPU, >=0 for CUDA device id)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, set_seed
    import torch

    # Prefer CPU by default; if CUDA available, pipeline will use GPU (faster but optional)
    device = 0 if torch.cuda.is_available() else -1

    try:
        # Attempt to load tokenizer & model (this may download weights on first run)
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
        # Log to console for debugging, but do not show stack trace to user
        print("Model load failed:", load_exc, file=sys.stderr)
        MODEL_AVAILABLE = False
except Exception as e:
    # transformers/torch not installed or import failed
    print("Transformers or torch import failed:", e, file=sys.stderr)
    MODEL_AVAILABLE = False

# --------------------------
# Fallback generator
# --------------------------
random.seed(FALLBACK_SEED)


def fallback_generate(prompt: str, max_tokens: int = 200, temperature: float = 0.7, style: str = "Friendly") -> str:
    """
    Deterministic-ish fallback generator:
    - Uses templates and small randomized choices for variety
    - Uses the last sentences from the prompt to craft a context-aware reply
    - Designed to be safe, quick, and informative when no HF model is available
    """
    # Basic templates per style
    templates = {
        "Friendly": [
            "Sure — here’s a helpful answer:\n\n{body}",
            "Okay! I think this will help:\n\n{body}"
        ],
        "Professional": [
            "Response (concise):\n\n{body}\n\nIf you need further details, ask a follow-up.",
            "Analysis:\n\n{body}\n\nRecommendation: validate with a small example."
        ],
        "Short": [
            "{body}",
            "Short answer: {body}"
        ],
        "Motivational": [
            "You’ve got this. Quick suggestion:\n\n{body}",
            "Motivational tip:\n\n{body}\n\nStart now and iterate."
        ],
    }

    body_candidates = []

    # Try to extract the user's last message or question
    try:
        last_user_line = prompt.strip().splitlines()[-1]
        # If it includes "User:" or "Assistant:" markers, strip them
        for prefix in ("User:", "Assistant:", "System:"):
            if last_user_line.startswith(prefix):
                last_user_line = last_user_line[len(prefix):].strip()
                break
    except Exception:
        last_user_line = ""

    # Create a few canned "helpful" bodies based on whether the last line looked like a question
    if "?" in last_user_line:
        body_candidates.append(
            f"It looks like you're asking about: \"{last_user_line}\". A good first step is to break the problem into smaller parts, try a minimal example, and test incrementally."
        )
        body_candidates.append(
            f"To address your question \"{last_user_line}\", try the following approach: clarify requirements, sketch a small reproducible example, and iterate."
        )
    else:
        body_candidates.append(
            "Thanks for the input — a recommended next step is to clarify the main goal, then create a minimal reproducible example to test."
        )
        body_candidates.append(
            "A practical suggestion: write a small test case that demonstrates the behavior you want, then expand from there."
        )

    # Add a canned helpful explanation that references prompt context
    if len(last_user_line) > 0:
        body_candidates.append(f"Based on what you wrote: \"{last_user_line}\", consider focusing on the core requirement and simplifying assumptions.")

    # choose a body and format with a template
    body = random.choice(body_candidates)
    template = random.choice(templates.get(style, templates["Friendly"]))
    reply = template.format(body=body)

    # Add a small deterministic pseudo-code or example if requested by content
    if "code" in prompt.lower() or "python" in prompt.lower():
        extra = "\n\nExample (pseudo):\n```\n# Start with a minimal example\nprint('hello world')\n# Expand step by step\n```\n"
        reply += extra

    # Respect max_tokens roughly by trimming
    if max_tokens and len(reply.split()) > max_tokens:
        reply_words = reply.split()[:max_tokens]
        reply = " ".join(reply_words) + "\n\n[truncated]"

    return reply


# --------------------------
# Helper: generate (model or fallback)
# --------------------------
def compose_prompt(history, system_instruction: str, style: str):
    """
    Build a text prompt for the model using the most recent CONTEXT_MESSAGES entries.
    History is a list of dicts with keys: role ('user'|'assistant'|'system') and 'text'
    """
    parts = []
    # include system instruction at top
    if system_instruction:
        parts.append(f"System: {system_instruction}")
    # pick last CONTEXT_MESSAGES messages
    recent = history[-CONTEXT_MESSAGES:] if len(history) >= CONTEXT_MESSAGES else history
    for m in recent:
        role_label = "User" if m["role"] == "user" else ("Assistant" if m["role"] == "assistant" else "System")
        parts.append(f"{role_label}: {m['text']}")
    # add assistant prefix to tell model to respond
    parts.append("Assistant:")
    prompt_text = "\n".join(parts)
    # Optionally prefix with a style hint
    if style == "Professional":
        prompt_text = "You are a concise professional assistant.\n" + prompt_text
    elif style == "Friendly":
        prompt_text = "You are a helpful and friendly assistant.\n" + prompt_text
    elif style == "Short":
        prompt_text = "You are brief and to the point.\n" + prompt_text
    elif style == "Motivational":
        prompt_text = "You are encouraging and motivational.\n" + prompt_text
    return prompt_text


def generate_response(history, system_instruction: str, user_message: str, max_tokens: int, temperature: float, style: str):
    """
    Use HF generator if available, otherwise fallback. Always return a string.
    """
    prompt = compose_prompt(history + [{"role": "user", "text": user_message}], system_instruction, style)

    # If model available, call pipeline with safety wrappers
    if MODEL_AVAILABLE and generator is not None:
        try:
            # set seed for reproducibility when desired
            try:
                set_seed(int(time.time()) % (2**31 - 1))
            except Exception:
                pass

            # Prepare generation parameters
            gen_kwargs = {
                "max_length": max_tokens + 50,  # rough cap (includes prompt tokens)
                "do_sample": True if temperature > 0 else False,
                "temperature": float(temperature),
                "num_return_sequences": 1,
            }
            # If the pipeline expects raw prompt, pass it
            input_text = prompt if INCLUDE_PROMPT_IN_GENERATION else user_message

            outputs = generator(input_text, **gen_kwargs)
            # pipeline returns list of dicts with 'generated_text'
            raw = outputs[0].get("generated_text", "")
            # Strip the prompt if present
            if raw.startswith(input_text):
                reply = raw[len(input_text):].strip()
            else:
                # Best effort to extract assistant reply
                reply = raw.strip()

            # Post-process: avoid returning empty string
            if not reply:
                reply = fallback_generate(prompt, max_tokens=max_tokens, temperature=temperature, style=style)
            # Safety: truncate extremely long responses
            if len(reply) > 20000:
                reply = reply[:20000] + "\n\n[truncated]"
            return reply
        except Exception as gen_exc:
            # Log internally but do not show traceback to user
            print("Model generation error:", gen_exc, file=sys.stderr)
            return fallback_generate(prompt, max_tokens=max_tokens, temperature=temperature, style=style)
    else:
        # Use fallback generator
        return fallback_generate(prompt, max_tokens=max_tokens, temperature=temperature, style=style)


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Local Streamlit Chatbot (no API key)", layout="wide")
st.title("Local Streamlit Chatbot — No API Key Required")

# Sidebar controls
with st.sidebar:
    st.header("Settings & Controls")
    system_instruction = st.text_area("System prompt (instructions for assistant)", value="You are a helpful assistant.", height=80)
    style = st.selectbox("Assistant style", options=["Friendly", "Professional", "Short", "Motivational"], index=0)
    max_tokens = st.slider("Max tokens / length (approx)", min_value=50, max_value=1024, value=200, step=50)
    temperature = st.slider("Temperature (creativity)", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
    st.markdown("---")
    st.write(f"Model status: **{'Loaded' if MODEL_AVAILABLE else 'Fallback (no model)'}**")
    if MODEL_AVAILABLE:
        st.write(f"- Model: `{MODEL_NAME}`")
        st.write(f"- Device: {'GPU' if device != -1 else 'CPU'}")
    st.markdown("---")
    if st.button("Clear chat history"):
        for k in list(st.session_state.keys()):
            # keep UI settings but remove chat history keys
            if k.startswith("chat_"):
                del st.session_state[k]
        st.session_state["chat_history"] = [
            {"role": "system", "text": system_instruction, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ]
        st.experimental_rerun()

    st.markdown("Export chat")
    if st.button("Prepare export"):
        # prepare export content
        hist = st.session_state.get("chat_history", [])
        export_text = ""
        for e in hist:
            export_text += f"[{e['time']}] {e['role'].upper()}: {e['text']}\n\n"
        st.session_state["export_text"] = export_text
        st.success("Chat prepared for download. Use the button below to download.")
    export_text = st.session_state.get("export_text", "")
    if export_text:
        st.download_button("Download chat (.txt)", data=export_text, file_name="chat_history.txt", mime="text/plain")

    st.markdown("---")
    st.markdown("Tips:")
    st.write("- If you have limited RAM or no internet, the fallback mode will be used automatically.")
    st.write("- To try a different HF model, edit MODEL_NAME at the top of app.py (small models recommended).")

# Initialize chat history in session_state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {"role": "system", "text": "You are a helpful assistant.", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    ]

# Layout: main chat area and side column for quick tools
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Conversation")
    # Show messages
    for msg in st.session_state.chat_history:
        role = msg["role"]
        ts = msg.get("time", "")
        if role == "user":
            st.markdown(f"**You — {ts}**")
            st.info(msg["text"])
        elif role == "assistant":
            st.markdown(f"**Assistant — {ts}**")
            st.write(msg["text"])
        else:
            st.markdown(f"**System — {ts}**")
            st.write(msg["text"])

    st.markdown("---")
    # Input form
    with st.form(key="user_input_form", clear_on_submit=False):
        user_input = st.text_area("Your message", height=140, placeholder="Type your message and press Send...")
        submitted = st.form_submit_button("Send")
        if submitted and user_input and user_input.strip():
            # Append user message
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append({"role": "user", "text": user_input.strip(), "time": now})
            # Generate assistant response
            try:
                with st.spinner("Generating response..."):
                    reply = generate_response(
                        history=st.session_state.chat_history,
                        system_instruction=system_instruction,
                        user_message=user_input.strip(),
                        max_tokens=max_tokens,
                        temperature=temperature,
                        style=style,
                    )
                    now2 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.chat_history.append({"role": "assistant", "text": reply, "time": now2})
            except Exception as e:
                # Log details to console for debugging, but show friendly message to user
                print("Unexpected error during generation:", e, file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                err_msg = "An unexpected error occurred while generating the response. Please try again or switch to fallback mode."
                st.session_state.chat_history.append({"role": "assistant", "text": err_msg, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            st.experimental_rerun()

with col2:
    st.subheader("Quick Prompts")
    if st.button("Example — Explain code"):
        st.session_state.chat_history.append({"role": "user", "text": "Explain this Python code and provide a simple usage example.", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        st.experimental_rerun()
    if st.button("Example — Summarize text"):
        st.session_state.chat_history.append({"role": "user", "text": "Summarize the following text into 5 bullet points: <paste text here>", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        st.experimental_rerun()
    if st.button("Example — Generate ideas"):
        st.session_state.chat_history.append({"role": "user", "text": "Give me 5 creative ideas for a short motivational video.", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Debug")
    st.write(f"Model loaded: **{MODEL_AVAILABLE}**")
    if MODEL_AVAILABLE:
        st.write(f"Model: `{MODEL_NAME}`")
        st.write(f"Device: {'GPU' if device != -1 else 'CPU'}")
    else:
        st.write("Using fallback generator (no model available).")

    if st.button("Show raw history"):
        st.json(st.session_state.chat_history)

# Footer note
st.markdown("---")
st.caption("This app runs fully locally (no API key). If you see long waits on first run, a model may be downloading. If you prefer not to download models, rely on fallback mode.")
