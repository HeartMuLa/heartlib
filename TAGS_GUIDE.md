# HeartMuLa - Tags Guide (Prompt Engineering)

This guide is based on the analysis of the HeartMuLa research paper (Sections 3.2 & 6.2). The model uses a natural language tokenizer (Llama 3) rather than a fixed dictionary. To achieve stable generation, select tags from the 8 primary categories used during training.

### The 8 Pillars of Training
Each category has an Importance percentage representing its "Selection Probability" during training.

* **Training Frequency:** Tags were "sampled" during training. Genre was included 95% of the time, while Instrument was only included 25%.
* **Model Expectations:** The model expects a Genre tag to function correctly. Without it, the generation lacks a clear structural anchor.
* **Influence vs. Stability:** Higher percentages equal higher stability. A 95% tag (Genre) is a "Strong Anchor," while a 10% tag (Topic) is a "Weak Hint" that may be ignored if it conflicts with stronger tags.
* **The Strategy:** For maximum control, lean heavily on the top 4 categories (Genre, Timbre, Gender, Mood). Use lower-percentage tags only as "seasoning" once the main structure is set.

### Official Categories

1. **GENRE** (95% - MANDATORY)
   Examples: Pop, Rock, Electronic, Hiphop, Jazz, Classical, Techno, Trance, Ambient.
2. **TIMBRE** (50% - Sound Texture)
   Examples: Soft, Warm, Husky, Bright, Dark, Distorted.
3. **GENDER** (37% - Vocal Character)
   Examples: Male, Female.
4. **MOOD** (32% - Emotional Vibe)
   Examples: Happy, Sad, Energetic, Joyful, Melancholic, Relaxing, Dark.
5. **INSTRUMENT** (25% - Dominant Sounds)
   Examples: Piano, Synthesizer, Acoustic Guitar, Electric Guitar, Bass, Drums, Strings, Violin.
6. **SCENE** (20% - Listening Context)
   Examples: Dance, Workout, Dating, Study, Cinematic, Party.
7. **REGION** (12% - Cultural Influence)
   Examples: K-pop, Latin, Western.
8. **TOPIC** (10% - Lyrical Theme)
   Examples: Love, Summer, Heartbreak.

### Prompting Strategy: "Less is More"
To maintain a strong anchor and avoid "Probability Interference," avoid conflicting tags.

* **Semantic Conflict:** Prompting "Rock, Jazz" splits the model's attention, often resulting in "muddy" or generic arrangements.
* **Anchor Stability:** One strong anchor provides a clear map. Multiple genres create conflicting maps, causing the AI to lose focus.
* **Recommendation:** Select only one tag per category. Be precise rather than broad.

### Recommended Format
Use a comma-separated list.

**Examples:**
* Electronic, Techno, Synthesizer, Dark, High Energy, Club
* Pop, Piano, Female, Sad, Soft, Love, Acoustic
