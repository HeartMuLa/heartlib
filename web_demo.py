from heartlib import HeartMuLaGenPipeline
import argparse
import torch
import gradio as gr
import tempfile
import os
import re
from google import genai
from google.genai import types
from openai import OpenAI

# Global pipeline
pipeline = None

# Check if LLM APIs are available
def check_llm_api_available():
    """Check if GEMINI_API_KEY or OPENAI_API_KEY is configured"""
    available_apis = []

    try:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key and gemini_key.strip():
            available_apis.append("gemini")
    except Exception:
        pass

    try:
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key and openai_key.strip():
            available_apis.append("openai")
    except Exception:
        pass

    return available_apis

# Default example from assets
EXAMPLE_LYRICS = """[Intro]

[Verse]
The sun creeps in across the floor
I hear the traffic outside the door
The coffee pot begins to hiss
It is another morning just like this

[Prechorus]
The world keeps spinning round and round
Feet are planted on the ground
I find my rhythm in the sound

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat
It is the ordinary magic that we meet

[Verse]
The hours tick deeply into noon
Chasing shadows,chasing the moon
Work is done and the lights go low
Watching the city start to glow

[Bridge]
It is not always easy,not always bright
Sometimes we wrestle with the night
But we make it to the morning light

[Chorus]
Every day the light returns
Every day the fire burns
We keep on walking down this street
Moving to the same steady beat

[Outro]
Just another day
Every single day"""

EXAMPLE_TAGS = "piano,happy"

# Tag categories for selection
TAG_DATA = {
    "Gender": [
        "Male", "Female"
    ],
    "Genre": [
        "Pop", "Folk", "Ballad", "Electronic", "Rock", "Acoustic", "R&B",
        "Indie", "Dance", "Indie Pop", "J-Pop", "Hip-Hop", "Country",
        "Latin", "Alternative", "Christian", "Cantopop", "Gospel", "Soul",
        "Mandopop"
    ],
    "Instrument": [
        "Drums", "Piano", "Guitar", "Strings", "Synthesizer", "Bass",
        "Acoustic Guitar", "Keyboard", "Electronic Drums", "Vocals",
        "Drum Machine", "Electric Guitar", "Percussion", "Beat",
        "Orchestra", "Saxophone", "Accordion", "Voice", "String", "Vocal"
    ],
    "Mood": [
        "Melancholy", "Romantic", "Energetic", "Hopeful", "Dreamy",
        "Relaxed", "Sad", "Calm", "Cheerful", "Reflective", "Emotional",
        "Joyful", "Sentimental", "Uplifting", "Warm", "Peaceful", "Upbeat",
        "Gentle", "Nostalgic", "Epic"
    ],
    "Scene": [
        "Driving", "Road Trip", "Cafe", "Relaxing", "Wedding", "Meditation",
        "Workout", "Walking", "Alone", "Travel", "Reflection", "Rainy Day",
        "Night", "Church", "Coffee Shop", "Gym", "Gaming", "Study",
        "Dating", "Date"
    ],
    "Singer Timbre": [
        "Soft", "Clear", "Warm", "Gentle", "Smooth", "Sweet", "Emotional",
        "Mellow", "Powerful", "Youthful", "Bright", "Rough", "Raspy",
        "Melodic", "Deep", "Soulful", "Strong", "Energetic", "Breathy",
        "Passionate"
    ],
    "Topic": [
        "Love", "Relationship", "Hope", "Longing", "Loss", "Heartbreak",
        "Memory", "Reflection", "Life", "Faith", "Regret", "Freedom",
        "Breakup", "Nature", "Loneliness", "Dreams", "Nostalgia", "Romance",
        "Friendship", "Youth"
    ]
}


def update_tag_string(*args):
    """
    Collects selected tags from all categories and joins them.
    args: list of lists (selections from each CheckboxGroup)
    """
    all_tags = []
    for selection in args:
        if selection:
            if isinstance(selection, list):
                all_tags.extend(selection)
            else:
                all_tags.append(selection)
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for t in all_tags:
        if t not in seen:
            unique_tags.append(t)
            seen.add(t)
    return ",".join(unique_tags)


def process_lyrics_correct(content):
    """
    Correct lyrics processing logic aligned with training data.
    1. Removes timestamps [xx:xx].
    2. Split lines and strip whitespace from each line.
    3. Remove leading/trailing empty lines.
    4. Collapse multiple newlines (3 or more) into 2.
    """
    # 0. Convert to lowercase
    content = content.lower()

    # 1. Remove timestamps [00:12] or [00:12.34]
    content = re.sub(r'\[[^\]]*\d{1,2}:\d{2}[^\]]*\]', '', content)

    # 2. Split lines and strip whitespace from each line
    lines = [line.strip() for line in content.split('\n')]

    # 3. Remove leading empty lines
    while lines and lines[0] == '':
        lines.pop(0)

    # 4. Remove trailing empty lines
    while lines and lines[-1] == '':
        lines.pop()

    # 5. Join back to string
    content = '\n'.join(lines)

    # 6. Collapse multiple newlines (3 or more) into 2
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content


def load_pipeline(model_path, version):
    """Load HeartMuLa pipeline"""
    global pipeline
    if pipeline is None:
        print(f"Loading model from {model_path}...")
        pipeline = HeartMuLaGenPipeline.from_pretrained(
            model_path,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
            version=version,
        )
        print("Model loaded!")
    return pipeline


def generate(lyrics, tags, cfg_scale, duration_sec, temperature, topk):
    """Generate music"""
    if pipeline is None:
        raise gr.Error("Model not loaded!")

    if not lyrics.strip():
        raise gr.Error("Please enter lyrics")

    if not tags.strip():
        raise gr.Error("Please enter tags")

    max_audio_length_ms = int(duration_sec * 1000)

    # Create temp file for output
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        output_path = f.name

    try:
        # Write lyrics and tags to temp files
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix=".txt", delete=False) as f:
            f.write(lyrics)
            lyrics_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix=".txt", delete=False) as f:
            f.write(tags)
            tags_path = f.name

        # Generate
        with torch.no_grad():
            pipeline(
                {
                    "lyrics": lyrics_path,
                    "tags": tags_path,
                },
                max_audio_length_ms=max_audio_length_ms,
                save_path=output_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )

        # Cleanup temp input files
        os.unlink(lyrics_path)
        os.unlink(tags_path)

        return output_path

    except Exception as e:
        raise gr.Error(f"Generation error: {str(e)}")


def generate_lyrics(theme, tags, language, api_choice, api_key_input, progress=gr.Progress()):
    """Generate lyrics using selected LLM API"""


    if not theme.strip():
        raise gr.Error("Please enter a theme")

    progress(0.1, desc="Preparing request...")

    # Determine API key
    api_key = api_key_input.strip() if api_key_input and api_key_input.strip() else None
    
    if not api_key:
        if api_choice == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
        elif api_choice == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            
    if not api_key:
        raise gr.Error(f"No API key provided for {api_choice}. Please enter an API key or set the environment variable.")


    # Language mapping
    language_names = {
        "en": "English",
        "zh": "Chinese",
        "jp": "Japanese",
        "kr": "Korean",
        "sp": "Spanish"
    }
    lang_name = language_names.get(language, "English")

    # Tags processing
    tags_text = tags.strip() if tags.strip() else "pop, emotional"

    # Create prompt
    prompt = f"""You are a professional songwriter. Generate song lyrics based on the following requirements:

**Theme**: {theme}
**Music Style/Tags**: {tags_text}
**Language**: {lang_name}

**Format Requirements** (CRITICAL):
1. Use lowercase for all lyrics text (except structure tags which are in brackets)
2. Include proper song structure tags: [Intro], [Verse], [Prechorus], [Chorus], [Bridge], [Outro]
3. Each structure tag should be on its own line
4. Separate different sections with a blank line (one empty line between sections)
5. NO timestamps like [00:12] - only structure tags allowed
6. Keep lyrics concise and suitable for a 3-4 minute song

**Structure Guidelines**:
- [Intro]: Optional, 1-2 lines if included
- [Verse]: Story-telling part, 4-6 lines, can repeat with different lyrics
- [Prechorus]: Optional, 2-4 lines, builds tension before chorus
- [Chorus]: Main hook, catchy and repetitive, 4-6 lines
- [Bridge]: Optional, provides contrast, 4-6 lines
- [Outro]: Closing, 1-2 lines

**Example Format**:
```
[Intro]

[Verse]
the sun creeps in across the floor
i hear the traffic outside the door
the coffee pot begins to hiss
it is another morning just like this

[Chorus]
every day the light returns
every day the fire burns
we keep on walking down this street
moving to the same steady beat
```

Now generate lyrics in {lang_name} based on the theme "{theme}" with style "{tags_text}".
Output ONLY the lyrics with structure tags, no explanations.
"""

    try:
        if api_choice == "gemini":
            # Gemini API
            progress(0.3, desc="Connecting to Gemini API...")

            # Set proxy if needed
            try:
                proxy_host = os.environ.get("PROXY_HOST", "127.0.0.1")
                proxy_port = os.environ.get("PROXY_PORT", "7890")
                os.environ['http_proxy'] = f'http://{proxy_host}:{proxy_port}'
                os.environ['https_proxy'] = f'http://{proxy_host}:{proxy_port}'
            except Exception:
                pass  # Proxy is optional

            # api_key is already set above
            client = genai.Client(api_key=api_key)

            progress(0.5, desc="Generating lyrics with Gemini...")

            response = client.models.generate_content(
                model='gemini-2.0-flash-lite',
                contents=[
                    types.Content(
                        role='user',
                        parts=[types.Part(text=prompt)]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.8,
                    max_output_tokens=2000
                )
            )

            generated_lyrics = response.text.strip()

        elif api_choice == "openai":
            # OpenAI API
            progress(0.3, desc="Connecting to OpenAI API...")

            # api_key is already set above
            client = OpenAI(api_key=api_key)

            progress(0.5, desc="Generating lyrics with OpenAI...")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional songwriter who creates well-structured lyrics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )

            generated_lyrics = response.choices[0].message.content.strip()

        else:
            raise gr.Error(f"Unknown API choice: {api_choice}")

        progress(0.9, desc="Processing response...")

        # Clean up the response (remove markdown code blocks if present)
        if generated_lyrics.startswith("```"):
            lines = generated_lyrics.split("\n")
            generated_lyrics = "\n".join(lines[1:-1]) if len(lines) > 2 else generated_lyrics

        # Apply our lyrics processing function to ensure format consistency
        generated_lyrics = process_lyrics_correct(generated_lyrics)

        progress(1.0, desc="Done!")

        return generated_lyrics

    except Exception as e:
        raise gr.Error(f"Lyrics generation error: {str(e)}")


def create_ui():
    """Create Gradio UI"""
    
    with gr.Blocks(title="HeartMuLa Music Generation") as demo:
        gr.Markdown("# HeartMuLa Music Generation")
        gr.Markdown("Generate music from lyrics and style tags")

        with gr.Tabs():
            # Tab 1: Music Generation
            with gr.Tab("Music Generation"):
                with gr.Row():
                    with gr.Column():
                        lyrics = gr.Textbox(
                            label="Lyrics",
                            lines=15,
                            value=EXAMPLE_LYRICS,
                            placeholder="Enter lyrics here..."
                        )

                        # Add format button
                        format_btn = gr.Button("Format Lyrics", size="sm")

                        # Tag Selection
                        gr.Markdown("### Tags")

                        tags = gr.Textbox(
                            label="Selected Tags (comma-separated)",
                            value=EXAMPLE_TAGS,
                            placeholder="e.g., piano,happy,pop",
                            lines=2
                        )

                        # Tag categories in accordion
                        tag_checkboxes = []
                        with gr.Accordion("Tag Categories (Click to Expand)", open=False):
                            with gr.Row():
                                with gr.Column():
                                    t1 = gr.CheckboxGroup(choices=TAG_DATA["Gender"], label="Gender")
                                    tag_checkboxes.append(t1)
                                    t2 = gr.CheckboxGroup(choices=TAG_DATA["Genre"], label="Genre")
                                    tag_checkboxes.append(t2)
                                with gr.Column():
                                    t3 = gr.CheckboxGroup(choices=TAG_DATA["Instrument"], label="Instrument")
                                    tag_checkboxes.append(t3)
                                    t4 = gr.CheckboxGroup(choices=TAG_DATA["Mood"], label="Mood")
                                    tag_checkboxes.append(t4)
                                with gr.Column():
                                    t5 = gr.CheckboxGroup(choices=TAG_DATA["Scene"], label="Scene")
                                    tag_checkboxes.append(t5)
                                    t6 = gr.CheckboxGroup(choices=TAG_DATA["Singer Timbre"], label="Singer Timbre")
                                    tag_checkboxes.append(t6)
                                with gr.Column():
                                    t7 = gr.CheckboxGroup(choices=TAG_DATA["Topic"], label="Topic")
                                    tag_checkboxes.append(t7)

                        # Generation parameters
                        with gr.Row():
                            cfg_scale = gr.Slider(0.0, 3.0, value=1.5, step=0.1, label="CFG Scale")
                            duration = gr.Slider(10, 300, value=180, step=10, label="Duration (sec)")

                        with gr.Row():
                            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                            topk = gr.Slider(1, 100, value=50, step=1, label="Top-K")

                        generate_btn = gr.Button("Generate Music", variant="primary", size="lg")

                    with gr.Column():
                        # Notice section
                        with gr.Accordion("Usage Notice", open=True):
                            gr.Markdown("""
### Lyrics Format Requirements

**Automatic Processing:**
1. All text will be converted to **lowercase**
2. Timestamps (e.g., [00:12]) will be **automatically removed**
3. Leading/trailing whitespace on each line will be **stripped**
4. Leading/trailing empty lines will be **removed**
5. Multiple consecutive empty lines (3+) will be **collapsed to 2**

**Recommended Format:**
- Use standard song structure tags: `[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`, etc.
- Separate sections with **blank lines**
- Case doesn't matter (will be auto-converted)

**Example:**
```
[Intro]

[Verse]
The sun creeps in across the floor
I hear the traffic outside the door

[Chorus]
Every day the light returns
Every day the fire burns
```

---

### Tags Format
- Use **commas** to separate multiple tags: `piano, happy, pop`
- Tags influence the style and mood of the generated music
- Select from categories below or type directly
                            """)

                        output_audio = gr.Audio(label="Generated Music", type="filepath")

                # Event handlers for tag selection
                for cb in tag_checkboxes:
                    cb.change(fn=update_tag_string, inputs=tag_checkboxes, outputs=tags)

                # Button callbacks
                format_btn.click(
                    fn=process_lyrics_correct,
                    inputs=[lyrics],
                    outputs=[lyrics]
                )

                generate_btn.click(
                    fn=generate,
                    inputs=[lyrics, tags, cfg_scale, duration, temperature, topk],
                    outputs=[output_audio]
                )

            # Tab 2: Lyrics Generation
            with gr.Tab("Lyrics Generation"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Generate Lyrics with AI")

                        api_selector = gr.Radio(
                            choices=[("Gemini", "gemini"), ("OpenAI", "openai")],
                            value="gemini",
                            label="Select LLM API"
                        )
                        
                        api_key_input = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="Enter your API Key here..."
                        )

                        theme_input = gr.Textbox(
                            label="Theme",
                            placeholder="e.g., Love lost in the city, Dreams and hope, Rainy day memories...",
                            lines=2
                        )

                        tags_gen = gr.Textbox(
                            label="Music Style/Tags",
                            placeholder="e.g., piano, melancholy, pop",
                            value="pop,emotional"
                        )

                        language_select = gr.Radio(
                            choices=[
                                ("English", "en"),
                                ("中文 (Chinese)", "zh"),
                                ("日本語 (Japanese)", "jp"),
                                ("한국어 (Korean)", "kr"),
                                ("Español (Spanish)", "sp")
                            ],
                            value="en",
                            label="Language"
                        )

                        generate_lyrics_btn = gr.Button(
                            "Generate Lyrics",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column():
                        with gr.Accordion("How to Use", open=True):
                            gr.Markdown("""
### Lyrics Generation Guide

**API Selection**: Choose between Gemini or OpenAI
- Gemini: Fast, good quality
- OpenAI: GPT-4o-mini, high quality

**Theme**: Describe the main idea or story you want in your song
- Examples: "A journey of self-discovery", "Missing someone far away", "Celebrating friendship"

**Music Style/Tags**: Specify the mood and genre
- Examples: "piano, sad, ballad", "guitar, upbeat, pop", "electronic, dreamy"

**Language**: Select the language for your lyrics
- Supported: English, Chinese, Japanese, Korean, Spanish

**Tips**:
- Be specific with your theme for better results
- Generated lyrics will follow the standard song structure
- You can edit the generated lyrics before using them for music generation
- The lyrics will be automatically formatted (lowercase, proper structure tags)
                            """)

                        generated_lyrics_output = gr.Textbox(
                            label="Generated Lyrics",
                            lines=20,
                            placeholder="Generated lyrics will appear here...",
                            interactive=False
                        )

                        copy_to_music_gen = gr.Button(
                            "Copy to Music Generation Tab",
                            size="sm"
                        )

                # Lyrics generation button callback
                generate_lyrics_btn.click(
                    fn=generate_lyrics,
                    inputs=[theme_input, tags_gen, language_select, api_selector, api_key_input],
                    outputs=[generated_lyrics_output]
                )

                # Copy lyrics to music generation tab
                def copy_lyrics(lyrics_text):
                    return lyrics_text

                copy_to_music_gen.click(
                    fn=copy_lyrics,
                    inputs=[generated_lyrics_output],
                    outputs=[lyrics]
                )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./ckpt")
    parser.add_argument("--version", type=str, default="3B")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("HeartMuLa Web Demo")
    print(f"Model: {args.model_path}")
    print(f"Version: {args.version}")

    # Load model
    load_pipeline(args.model_path, args.version)

    # Launch UI
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
