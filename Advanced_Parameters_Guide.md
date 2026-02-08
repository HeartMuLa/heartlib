# HeartMuLa - Advanced Parameters Guide

**Disclaimer:**
> The **Default** values listed below are explicitly cited from the official HeartMuLa research paper. The paper does not specify absolute minimum or maximum limits.
> The *adjustments* and "recipes" are derived from standard Large Language Model (LLM) behavior and community experimentation. While HeartMuLa is based on Llama-3.2, results may vary depending on the specific model checkpoint.

---

## The "Mellow Bias": Why your Techno sounds like Pop
Many users report that the model ignores aggressive tags (e.g., *Techno, Metal, High Energy*) and defaults to a "mellow" or "pop" sound.

**The Likely Cause:**
The HeartMuLa training dataset was filtered using **Audiobox-Aesthetic** scores to ensure high fidelity.
* **Fact:** Aesthetic filters are trained to prefer "clean" audio.
* **The Side Effect:** In generative audio, "clean" often correlates with Pop, Acoustic, or Soft textures, while "distorted" or "noisy" genres (like Dubstep or Rock) can be penalized.
* **The Result:** When the model is unsure, it drifts toward the "safe" aesthetic (Mellow/Pop).

To break this bias, you must adjust the **Inference Parameters** to force the model away from its "safe" center.

---

## 1. Classifier-Free Guidance (`--cfg_scale`)
**The "Strictness" Slider.**
This parameter controls how strongly the model forces the audio to match your text prompt (Tags/Lyrics) versus its internal training distribution.

* **Paper Default:** `1.5`
* **How it works (General AI Logic):**
    * **Low (1.0 - 1.5):** The model is "loosely" guided by your tags. It prioritizes the internal "aesthetic" bias (smooth/safe audio).
    * **High (2.0 - 4.0):** The model is "forced" to match your tags, even if the result is less "aesthetically safe."

**Community Observation:**
Users have reported that the default `1.5` is often too weak for specific genres. Increasing this value has helped users generate genres like R&B that were previously generating as Pop.

**Recommendation:**
If your genre is being ignored, **increase** this value. Start with `2.5` and go up to `4.0` if needed.

---

## 2. Temperature (`--temperature`)
**The "Randomness" Slider.**
This controls the probability distribution for the next audio token.

* **Paper Default:** `1.0`
* **How it works (General AI Logic):**
    * **Lower (< 1.0):** The model becomes "conservative." It picks only the most likely sounds. This usually results in more repetitive, structured, and coherent rhythms.
    * **Higher (> 1.0):** The model becomes "creative" and takes risks. This adds variety but increases the chance of chaos or the melody falling apart.

**Recommendation:**
For genres requiring strict rhythm (Techno, House), try **lowering** this slightly to `0.8` or `0.9` to lock in the groove.

---

## 3. Top-K (`--topk`)
**The "Vocabulary" Limit.**
This limits the pool of possible "next sounds" to the top *K* most likely options.

* **Paper Default:** `50`
* **How it works:**
    * A standard setting for Llama-based models. Lowering this (e.g., to 30) can reduce "hallucinations" or random artifacts, but may make the audio sound dull.

---

## 🧪 Experimental "Recipes"

These settings are suggestions based on how Autoregressive Transformers generally respond to these parameters. They are not official presets.

### A. The "Aggressive / Specific" Fix
*Use this if the model is ignoring your Genre tags (e.g., Metal, Techno, Rap).*
* **Logic:** High CFG forces the genre; Lower Temp keeps the rhythm tight.
* **Command:**
    `--cfg_scale 3.0 --temperature 0.8`

### B. The "Creative / Jazz" Flow
*Use this for genres that benefit from improvisation or loose timing.*
* **Logic:** Moderate CFG allows some freedom; Higher Temp encourages unique melodies.
* **Command:**
    `--cfg_scale 2.0 --temperature 1.1`

### C. The "Safe / High Fidelity" (Paper Default)
*Use this for Pop, Ballads, or when audio quality is the priority.*
* **Logic:** Low CFG prioritizes the "Aesthetic" filter; Default Temp ensures standard variety.
* **Command:**
    `--cfg_scale 1.5 --temperature 1.0`
