"""LLM generator for lyrics and tags using local Transformers runtime.

Supports CUDA, Intel XPU, and CPU-only modes through PyTorch device selection.
"""
from typing import Dict, Tuple, Optional
import os
import re
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Model selection (user requested)
# You can override this with HEARTLIB_LLM_MODEL if needed.
DEFAULT_MODEL = os.getenv("HEARTLIB_LLM_MODEL", "Qwen/Qwen3-0.6B")

# Simple parser for the expected model output
_OUTPUT_SECTION_RE = re.compile(r"(?m)^(LYRICS:|TAGS:)\s*$", re.IGNORECASE)


class LLMGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[torch.device] = None):
        self.model_name = model_name
        self.requested_device = str(device) if device is not None else "auto"
        # allow explicit device override (string or torch.device)
        if isinstance(device, str) and device.lower() != "auto":
            self.device = torch.device(device)
        else:
            self.device = device or self._detect_device()
        self._load()

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        # XPU (Intel) support — prefer running directly on xpu when available
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    def _load(self):
        # Guard against accidental VLM IDs (e.g., Qwen3-VL) that are incompatible
        # with AutoModelForCausalLM in this text-only pipeline.
        try:
            cfg = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            model_type = str(getattr(cfg, "model_type", "")).lower()
            cfg_name = cfg.__class__.__name__.lower()
            if "vl" in model_type or "vl" in cfg_name:
                fallback_model = "Qwen/Qwen3-0.6B"
                print(
                    f"[LLM][WARN] Model '{self.model_name}' is vision-language ({cfg.__class__.__name__}). "
                    f"Falling back to text model '{fallback_model}'."
                )
                self.model_name = fallback_model
        except Exception:
            # Continue with the configured model name if config probing fails.
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        if self.device.type == "cuda":
            target_dtype = torch.float16
        elif self.device.type == "xpu":
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=target_dtype,
                trust_remote_code=True,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

        try:
            self.model.to(self.device)
        except Exception:
            self.device = torch.device("cpu")
            self.model.to(self.device)

        self.model.eval()

    def _length_hint(self, length: str) -> str:
        mapping = {
            "short": "8-12 lines",
            "medium": "16-24 lines",
            "long": "28-40 lines",
        }
        return mapping.get((length or "").lower(), "16-24 lines")

    def _format_prompt(self, style: str, length: str, include_structure: bool) -> str:
        structure_note = (
            "Include structure tags like [Verse], [Chorus], [Bridge] in the lyrics." if include_structure else "Do not include structure tags."
        )
        target_len = self._length_hint(length)
        prompt = (
            "You are a music lyric generator. Follow instructions exactly.\n"
            f"STYLE: {style}\n"
            f"TARGET LENGTH: {target_len}\n"
            f"STRUCTURE: {structure_note}\n\n"
            "MANDATORY RULES:\n"
            "1) Output must have exactly two sections in this exact order:\n"
            "LYRICS:\n"
            "<lyrics content>\n\n"
            "TAGS:\n"
            "tag1,tag2,tag3\n"
            "2) If structure is requested, each section header must be followed by at least one lyric line.\n"
            "3) Do not output only headers.\n"
            "4) Do not include explanations, analysis, thoughts, markdown fences, JSON, or extra text.\n"
            "5) TAGS must be lowercase, comma-separated, no spaces after commas.\n"
            "6) Do NOT output placeholder tags like tag1,tag2,tag3 or ....\n"
            "7) Generate 5-8 meaningful tags based on style/mood/instrument/era (example: synthwave,retro,electronic,neon,80s).\n"
        )
        return prompt

    def _strip_lyrics_wrappers(self, lyrics: str) -> str:
        cleaned = (lyrics or "").strip()
        cleaned = re.sub(r"(?is)^<lyrics>\s*", "", cleaned)
        cleaned = re.sub(r"(?is)\s*</lyrics>$", "", cleaned)
        return cleaned.strip()

    def _is_placeholder_tags(self, tags: str) -> bool:
        normalized = (tags or "").strip().lower().replace(" ", "")
        if not normalized:
            return True
        if normalized in {"tag1,tag2,tag3", "tag1,tag2,...", "tag1,tag2", "..."}:
            return True
        return bool(re.fullmatch(r"tag\d+(,tag\d+){1,10}", normalized))

    def _is_structure_headers_only(self, lyrics: str) -> bool:
        text = (lyrics or "").strip()
        if not text:
            return False
        # every non-empty line must be a bracketed section header
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        return all(re.fullmatch(r"\[[^\]]+\]", line) for line in lines)

    def _is_header_line(self, line: str) -> bool:
        return bool(re.fullmatch(r"\[(verse(?:\s*\d+)?|chorus(?:\s*\d+)?|bridge|pre[- ]?chorus|intro|outro)\]", line.strip(), re.IGNORECASE))

    def _normalize_header(self, line: str) -> str:
        stripped = line.strip()
        m = re.fullmatch(r"\[\s*([^\]]+?)\s*\]", stripped)
        if not m:
            return stripped
        header = m.group(1).strip().lower()
        if header.startswith("verse"):
            num = re.search(r"\d+", header)
            return f"[Verse {num.group(0)}]" if num else "[Verse]"
        if header.startswith("chorus"):
            num = re.search(r"\d+", header)
            return f"[Chorus {num.group(0)}]" if num else "[Chorus]"
        if header in {"bridge", "intro", "outro"}:
            return f"[{header.title()}]"
        if header in {"pre-chorus", "pre chorus", "prechorus"}:
            return "[Pre-Chorus]"
        return stripped

    def _is_tag_like_line(self, line: str) -> bool:
        candidate = line.strip().lower()
        if not candidate:
            return False
        if re.fullmatch(r"[a-z0-9\- ]+(,\s*[a-z0-9\- ]+){2,}", candidate):
            words = [token.strip() for token in candidate.split(",") if token.strip()]
            # tag-like lists are usually short tokens with no sentence punctuation
            if words and all(len(token.split()) <= 3 for token in words):
                return True
        return False

    def _sanitize_structured_lyrics(self, lyrics: str) -> str:
        lines = [line.rstrip() for line in (lyrics or "").splitlines()]
        output: list[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if output and output[-1] != "":
                    output.append("")
                continue

            if self._is_header_line(line):
                normalized = self._normalize_header(line)
                if output and output[-1] != "":
                    output.append("")
                output.append(normalized)
                continue

            # remove inline bracket tags from lyric lines (e.g. ", [bridge]")
            line = re.sub(r"\s*\[[^\]]+\]\s*", " ", line)
            line = re.sub(r"\s+", " ", line).strip(" ,")
            if not line:
                continue
            if self._is_tag_like_line(line):
                continue

            output.append(line)

        # collapse repeated blank lines
        cleaned: list[str] = []
        for line in output:
            if line == "" and cleaned and cleaned[-1] == "":
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _is_low_quality_structured(self, lyrics: str) -> bool:
        lines = [line.strip() for line in (lyrics or "").splitlines() if line.strip()]
        if not lines:
            return True
        non_header = [line for line in lines if not self._is_header_line(line)]
        if len(non_header) < 6:
            return True
        if any(re.search(r"\[[^\]]+\]", line) for line in non_header):
            return True
        unique_ratio = len(set(non_header)) / max(1, len(non_header))
        if unique_ratio < 0.6:
            return True
        return False

    def _rewrite_structured_lyrics(self, draft_lyrics: str, style: str, length: str, seed: Optional[int], max_new_tokens: int) -> str:
        rewrite_prompt = (
            f"Rewrite these lyrics in style '{style}' and target length '{length}'.\n"
            "STRICT FORMAT RULES:\n"
            "- Use section headers on their own lines only: [Verse], [Chorus], [Bridge], [Intro], [Outro].\n"
            "- Never place bracketed labels inside lyric lines.\n"
            "- Under each section, write 2-4 full lyric lines.\n"
            "- Avoid repeating the exact same lyric line multiple times.\n"
            "- Return only cleaned lyrics, no TAGS, no explanations.\n\n"
            f"DRAFT:\n{draft_lyrics.strip()}"
        )
        return self._model_generate(
            prompt=rewrite_prompt,
            temperature=0.62,
            top_p=0.82,
            seed=seed,
            max_new_tokens=max(300, max_new_tokens),
        )

    def _repair_structure_only(self, lyrics: str, style: str, length: str, seed: Optional[int], max_new_tokens: int) -> str:
        repair_prompt = (
            f"Expand these section headers into real song lyrics in style '{style}' with target length '{length}'.\n"
            "Keep the same headers and add 2-4 lyric lines under each header.\n"
            "Return only lyrics, no TAGS section, no explanations.\n\n"
            f"HEADERS:\n{lyrics.strip()}"
        )
        return self._model_generate(
            prompt=repair_prompt,
            temperature=0.65,
            top_p=0.85,
            seed=seed,
            max_new_tokens=max(256, max_new_tokens),
        )

    def _model_generate(self, prompt: str, temperature: float, top_p: float, seed: Optional[int], max_new_tokens: int) -> str:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Use Qwen chat template when available; disable thinking mode for concise
        # deterministic structure following as suggested in model docs.
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            chat_text = prompt

        inputs = self.tokenizer(chat_text, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=20,
                        repetition_penalty=1.15,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            elif self.device.type == "xpu":
                try:
                    with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                        out = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=float(temperature),
                            top_p=float(top_p),
                            top_k=20,
                            repetition_penalty=1.15,
                            do_sample=True,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                except Exception:
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        top_k=20,
                        repetition_penalty=1.15,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=20,
                    repetition_penalty=1.15,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        generated_tokens = out[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return text

    def generate_lyrics_and_tags(
        self,
        style: str = "pop",
        length: str = "medium",
        include_structure: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        top_p: float = 0.95,
        seed: Optional[int] = None,
    ) -> Dict[str, str]:
        prompt = self._format_prompt(style, length, include_structure)
        text = ""
        lyrics = tags = ""

        for attempt in range(2):
            text = self._model_generate(
                prompt=prompt,
                temperature=(temperature if attempt == 0 else max(0.2, temperature - 0.2)),
                top_p=top_p,
                seed=(None if seed is None else seed + attempt),
                max_new_tokens=(max_new_tokens if attempt == 0 else max_new_tokens + 128),
            )
            lyrics, tags = self._parse_output(text)
            lyrics = self._strip_lyrics_wrappers(lyrics)
            if self._is_placeholder_tags(tags):
                tags = ""

            if include_structure and self._is_structure_headers_only(lyrics):
                repaired = self._repair_structure_only(
                    lyrics=lyrics,
                    style=style,
                    length=length,
                    seed=(None if seed is None else seed + attempt + 100),
                    max_new_tokens=max_new_tokens,
                )
                repaired_lyrics, _ = self._parse_output(repaired)
                repaired_lyrics = self._strip_lyrics_wrappers(repaired_lyrics or repaired)
                if repaired_lyrics and not self._is_structure_headers_only(repaired_lyrics):
                    lyrics = repaired_lyrics

            if include_structure and lyrics:
                lyrics = self._sanitize_structured_lyrics(lyrics)
                if self._is_low_quality_structured(lyrics):
                    rewritten = self._rewrite_structured_lyrics(
                        draft_lyrics=lyrics,
                        style=style,
                        length=length,
                        seed=(None if seed is None else seed + attempt + 200),
                        max_new_tokens=max_new_tokens,
                    )
                    rewritten_lyrics, _ = self._parse_output(rewritten)
                    rewritten_lyrics = self._sanitize_structured_lyrics(self._strip_lyrics_wrappers(rewritten_lyrics or rewritten))
                    if rewritten_lyrics:
                        lyrics = rewritten_lyrics

            marker_only = bool(re.match(r"^\s*(\[[A-Za-z0-9 _\-]+\]\s*(,|\n)?\s*)+$", (lyrics or text).strip()))
            if lyrics and not marker_only:
                break

        if not lyrics and text:
            lyrics = self._strip_lyrics_wrappers(text)
        return {"lyrics": lyrics, "tags": tags, "raw": text}

    def _parse_output(self, text: str) -> Tuple[str, str]:
        # robust splitter: prefer explicit LYRICS:/TAGS: markers but fall back to
        # simple heuristics when markers are missing.
        text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
        text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text.strip())
        lyrics = ""
        tags = ""

        # Try explicit markers first (case-insensitive, dotall)
        m_lyrics = re.search(r"(?is)LYRICS:\s*(.*?)\s*(?:TAGS:|$)", text)
        m_tags = re.search(r"(?is)TAGS:\s*(.*)$", text)
        if m_lyrics:
            lyrics = m_lyrics.group(1).strip()
        if m_tags:
            tags = m_tags.group(1).strip()

        # If no explicit markers, attempt to split by a blank line followed by short CSV-like line
        if not (lyrics or tags):
            parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if len(parts) >= 2 and re.match(r"^[a-zA-Z0-9,\- ]+$", parts[-1]):
                # last paragraph looks like tags
                tags = parts[-1]
                lyrics = "\n\n".join(parts[:-1])
            else:
                # treat everything as lyrics
                lyrics = text.strip()

        # postprocess tags into comma-separated short tokens
        tags = tags.replace("\n", " ")
        tags = re.sub(r"\s*,\s*", ",", tags)
        tags = re.sub(r"[^a-zA-Z0-9,\- ]", "", tags)
        tags = ",".join([t.strip().lower() for t in tags.split(",") if t.strip()])

        return self._strip_lyrics_wrappers(lyrics.strip()), tags


# convenience function for quick usage
_generator_singleton: Optional[LLMGenerator] = None


def get_generator(model_name: str = DEFAULT_MODEL, device: Optional[str] = None) -> LLMGenerator:
    """Return a singleton LLMGenerator. Pass `device` as one of: `None`/"auto"/"cuda"/"cpu"/"cuda:0` etc.
    When `device` differs from the existing singleton, the generator will be re-created."""
    global _generator_singleton
    if _generator_singleton is None or _generator_singleton.model_name != model_name or getattr(_generator_singleton, "requested_device", "auto") != (str(device) if device is not None else "auto"):
        _generator_singleton = LLMGenerator(model_name=model_name, device=device)
    return _generator_singleton


if __name__ == "__main__":
    g = get_generator()
    out = g.generate_lyrics_and_tags(style="synthwave", length="short", include_structure=True, seed=42)
    print("=== LYRICS ===")
    print(out['lyrics'])
    print("=== TAGS ===")
    print(out['tags'])
