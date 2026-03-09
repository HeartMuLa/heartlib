"""Standalone Gradio app to generate lyrics & tags using a lightweight LLM.

This file intentionally does not modify existing project files. It provides a small UI
that you can run alongside the main `webui.py` and copy results into the main app.
"""
import gradio as gr
import re
import os
import sys

# Ensure local `src` package is imported before any installed `site-packages` copy.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from heartlib.llm.generator import get_generator

# Use the same Origin theme as the main WebUI so appearance is consistent.
APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui"],
).set(
    body_background_fill="*neutral_50",
    block_title_text_weight="600",
    block_border_width="1px",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

GEN = None


def _get_gen(device: str | None = None):
    """Return singleton generator; pass `device` through to `get_generator`.
    `get_generator` is responsible for recreating the singleton only when needed."""
    global GEN
    if device:
        # delegate decision to get_generator (it will reuse the singleton when appropriate)
        GEN = get_generator(device=device)
    elif GEN is None:
        GEN = get_generator()
    return GEN


def generate(style, length, include_structure, temperature, top_p, seed, device):
    """Safe wrapper around the generator invoked by the Gradio button.
    - logs start/finish to console so you see progress
    - normalizes seed input
    - returns error text into the `raw` output if generation fails
    """
    print(f"[LLM] generate() called — style={style!r}, length={length}, seed={seed}")
    gen = _get_gen(device=device)

    # normalize seed (gradio may pass None or a float)
    if seed is None or seed == "":
        seed_val = None
    else:
        try:
            seed_val = int(seed)
        except Exception:
            seed_val = None

    try:
        out = gen.generate_lyrics_and_tags(
            style=style,
            length=length,
            include_structure=include_structure,
            temperature=float(temperature),
            top_p=float(top_p),
            seed=seed_val,
        )
        print("[LLM] generation complete")

        # Post-process outputs: prefer structured parsing for tags and
        # structural markers so the UI fields are populated sensibly.
        lyrics = out.get("lyrics", "") or ""
        tags = out.get("tags", "") or ""
        raw = out.get("raw", "") or ""

        # 1) If the model returned an explicit TAGS: section, rely on parser.
        # 2) If parser produced nothing but raw looks like a CSV tag list
        #    (e.g. "tag1,tag2" or "[tag1,tag2]") -> populate `tags_out`.
        tag_like_bracket = re.match(r"^\s*\[\s*[A-Za-z0-9\- ]+(?:\s*,\s*[A-Za-z0-9\- ]+)+\s*\]\s*$", raw)
        tag_like_plain = re.match(r"^\s*[A-Za-z0-9\- ]+(?:\s*,\s*[A-Za-z0-9\- ]+)+\s*$", raw)
        # Search for a TAGS: line anywhere in the raw output (not only at the start).
        explicit_tags_prefix = re.search(r"(?is)TAGS:\s*(.*)$", raw)

        if not tags and raw and (tag_like_bracket or tag_like_plain or explicit_tags_prefix):
            # extract inner CSV (strip brackets / 'TAGS:' if present)
            inner = explicit_tags_prefix.group(1) if explicit_tags_prefix else re.sub(r"^\s*\[|\]\s*$", "", raw).strip()
            # normalize into comma-separated, lowercase tokens with no spaces
            inner = inner.replace("\n", " ")
            inner = re.sub(r"\s*,\s*", ",", inner)
            inner = re.sub(r"[^a-zA-Z0-9,\- ]", "", inner)
            tags = ",".join([t.strip().lower() for t in inner.split(",") if t.strip()])

        # 3) If parser returned nothing and raw looks like structural markers
        #    such as [Verse], [Chorus], show them in the Lyrics box.
        if not lyrics and raw and re.match(r"^\s*(\[[A-Za-z0-9 _\-]+\]\s*(,|\n)?\s*)+$", raw):
            # but avoid treating tag-like single-bracket CSV as structure
            if not (tag_like_bracket or tag_like_plain or explicit_tags_prefix):
                formatted = re.sub(r"\s*,\s*", "\n\n", raw.strip())
                lyrics = formatted

        return lyrics, tags, raw
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[LLM][ERROR] generation failed:\n{tb}")
        err_msg = f"ERROR: {e}\nSee server logs for details."
        return "", "", err_msg


with gr.Blocks(title="HeartMuLa — LLM Lyrics & Tags Generator", theme=APP_THEME) as demo:
    gr.Markdown("# ✨ LLM Lyrics & Tags Generator (Qwen/Qwen3-0.6B, local runtime)")
    with gr.Row():
        with gr.Column(scale=2):
            style = gr.Textbox(label="Style / Prompt", value="electronic synthwave / 80s")
            length = gr.Dropdown(label="Length", choices=["short", "medium", "long"], value="medium")
            include_structure = gr.Checkbox(label="Include structure tags ([Verse], [Chorus])", value=True)
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, value=0.7, step=0.05)
            top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=0.8, step=0.01)
            seed = gr.Textbox(label="Seed (optional, integer)", value="")
            device = gr.Dropdown(label="Device", choices=["auto", "cuda", "xpu", "cpu"], value="auto", info="Select execution device. 'auto' will prefer CUDA then XPU if available.")
            gen_btn = gr.Button("Generate")
        with gr.Column(scale=3):
            lyrics_out = gr.Textbox(label="Generated Lyrics", lines=14)
            tags_out = gr.Textbox(label="Generated Tags (comma-separated)")
            raw_out = gr.Textbox(label="Raw model output", lines=6)

    gen_btn.click(fn=generate, inputs=[style, length, include_structure, temperature, top_p, seed, device], outputs=[lyrics_out, tags_out, raw_out])

    gr.Markdown(
        "Usage: run `python webui_llm.py` and copy results into the main WebUI fields.\n"
        "Default model: `Qwen/Qwen3-0.6B` (override via `HEARTLIB_LLM_MODEL` if needed)."
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
