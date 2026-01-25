import os
import gradio as gr
import torch
from qwen_tts import Qwen3TTSModel

# Cache models in-memory so switching tabs is fast
MODELS = {}

HF_HOME = os.environ.get("HF_HOME", "/data/hf")
os.environ["HF_HOME"] = HF_HOME

# If you don't have flash-attn installed, use SDPA to avoid ImportError.
# If you DO install flash-attn, you can set ATTENTION_IMPL=flash_attention_2.
ATTN_IMPL = os.environ.get("ATTENTION_IMPL", "sdpa")  # "sdpa" or "flash_attention_2"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE_MAP = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_model(model_id: str) -> Qwen3TTSModel:
    if model_id not in MODELS:
        MODELS[model_id] = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=DEVICE_MAP,
            dtype=DTYPE,
            attn_implementation=ATTN_IMPL,
        )
    return MODELS[model_id]

# Supported languages per README (10 major languages) + Auto for convenience.
LANG_CHOICES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# CustomVoice speakers + descriptions + native language from README table.
SPEAKER_INFO = {
    "Vivian":     {"desc": "Bright, slightly edgy young female voice.", "native": "Chinese"},
    "Serena":     {"desc": "Warm, gentle young female voice.",          "native": "Chinese"},
    "Uncle_Fu":   {"desc": "Seasoned male voice with a low, mellow timbre.", "native": "Chinese"},
    "Dylan":      {"desc": "Youthful Beijing male voice with a clear, natural timbre.", "native": "Chinese (Beijing Dialect)"},
    "Eric":       {"desc": "Lively Chengdu male voice with a slightly husky brightness.", "native": "Chinese (Sichuan Dialect)"},
    "Ryan":       {"desc": "Dynamic male voice with strong rhythmic drive.", "native": "English"},
    "Aiden":      {"desc": "Sunny American male voice with a clear midrange.", "native": "English"},
    "Ono_Anna":   {"desc": "Playful Japanese female voice with a light, nimble timbre.", "native": "Japanese"},
    "Sohee":      {"desc": "Warm Korean female voice with rich emotion.", "native": "Korean"},
}
SPEAKER_CHOICES = list(SPEAKER_INFO.keys())

def speaker_help_md(speaker: str) -> str:
    info = SPEAKER_INFO.get(speaker)
    if not info:
        return ""
    return (
        f"**Speaker:** `{speaker}`  \n"
        f"**Voice description:** {info['desc']}  \n"
        f"**Native language (recommended):** {info['native']}  \n\n"
        f"> Note: The README recommends using each speaker’s native language for best quality; "
        f"speakers can still speak any supported language."
    )

def maybe_apply_native_language(selected_language: str, speaker: str, use_native: bool) -> str:
    if not use_native:
        return selected_language
    native = SPEAKER_INFO.get(speaker, {}).get("native", "")
    # Map dialect label to base language dropdown where possible
    if native.startswith("Chinese"):
        return "Chinese"
    if native.startswith("English"):
        return "English"
    if native.startswith("Japanese"):
        return "Japanese"
    if native.startswith("Korean"):
        return "Korean"
    return selected_language

def customvoice_tts(text, language, speaker, instruct, use_native_lang):
    model = get_model("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    language = maybe_apply_native_language(language, speaker, use_native_lang)

    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language if language else "Auto",
        speaker=speaker,
        instruct=instruct or "",
    )
    return (sr, wavs[0])

def voicedesign_tts(text, language, instruct):
    model = get_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    wavs, sr = model.generate_voice_design(
        text=text,
        language=language if language else "Auto",
        instruct=instruct or "",
    )
    return (sr, wavs[0])

def voiceclone_tts(text, language, ref_audio, ref_text, x_vector_only_mode):
    model = get_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language if language else "Auto",
        ref_audio=ref_audio,
        ref_text=ref_text if (ref_text and not x_vector_only_mode) else None,
        x_vector_only_mode=bool(x_vector_only_mode),
    )
    return (sr, wavs[0])

with gr.Blocks(title="Qwen3-TTS Local Tabs") as demo:
    gr.Markdown(
        "## Qwen3-TTS (Local) — Tabs like the HF Space\n\n"
        "- **Languages:** Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian (+ `Auto`).\n"
        "- **CustomVoice:** preset speakers; native language is recommended for best quality.\n"
        "- **Voice Clone:** best quality when you provide both **reference audio** and its **transcript**; "
        "`x_vector_only_mode` can work without transcript but may reduce quality.\n"
    )

    with gr.Tabs():

        with gr.Tab("TTS (CustomVoice)"):
            gr.Markdown(
                "### CustomVoice\n"
                "Choose a preset speaker and (optionally) an instruction (emotion, pace, style).\n"
            )

            t = gr.Textbox(label="Text", lines=4)
            lang = gr.Dropdown(choices=LANG_CHOICES, value="Auto", label="Language")
            speaker = gr.Dropdown(choices=SPEAKER_CHOICES, value="Vivian", label="Speaker")
            speaker_info = gr.Markdown(speaker_help_md("Vivian"))
            use_native = gr.Checkbox(value=True, label="Use speaker's native language (recommended)")
            instr = gr.Textbox(label="Style / Instruct (optional)", lines=2, placeholder="e.g., Very happy. / Speak slowly and calmly.")

            out = gr.Audio(label="Output", type="numpy")
            btn = gr.Button("Generate")

            speaker.change(fn=speaker_help_md, inputs=speaker, outputs=speaker_info)
            btn.click(customvoice_tts, [t, lang, speaker, instr, use_native], out)

        with gr.Tab("Voice Design"):
            gr.Markdown(
                "### Voice Design\n"
                "Describe the voice in natural language (age, gender, timbre, mood, speaking style), "
                "then generate speech in that designed voice.\n"
            )
            t = gr.Textbox(label="Text", lines=4)
            lang = gr.Dropdown(choices=LANG_CHOICES, value="Auto", label="Language")
            instr = gr.Textbox(
                label="Voice description / Instruct",
                lines=4,
                placeholder="e.g., Male, 17 years old, tenor range, slightly nervous but gaining confidence..."
            )
            out = gr.Audio(label="Output", type="numpy")
            btn = gr.Button("Generate")
            btn.click(voicedesign_tts, [t, lang, instr], out)

        with gr.Tab("Voice Clone (Base)"):
            gr.Markdown(
                "### Voice Clone (Base)\n"
                "- Provide **reference audio** + **reference transcript** for best cloning quality.\n"
                "- If you enable `x_vector_only_mode`, transcript is optional, but quality may drop.\n"
            )
            t = gr.Textbox(label="Target text", lines=4)
            lang = gr.Dropdown(choices=LANG_CHOICES, value="Auto", label="Language")
            ref_audio = gr.Audio(label="Reference audio (upload)", type="filepath")
            ref_text = gr.Textbox(label="Reference transcript (recommended)", lines=3)
            xvec = gr.Checkbox(label="x_vector_only_mode (no ref_text; may reduce quality)", value=False)

            out = gr.Audio(label="Output", type="numpy")
            btn = gr.Button("Generate")
            btn.click(voiceclone_tts, [t, lang, ref_audio, ref_text, xvec], out)

demo.launch(server_name="0.0.0.0", server_port=8000)
