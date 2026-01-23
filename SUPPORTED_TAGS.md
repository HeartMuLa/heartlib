# Supported Tags for HeartMuLa

This document lists tags that have been tested and shown to have some effect on music generation. The 3B model has limited tag conditioning, so results may vary.

## Tips for Better Results

1. **Use CFG scale 2.0-2.5** for stronger tag following (default is now 2.5)
2. **Put style descriptions in lyrics** if tags alone don't work - the model responds better to natural language
3. **Format:** `tag1,tag2,tag3` (comma-separated, lowercase)
4. **Keep it simple:** 3-5 tags work better than many tags
5. **Unknown tags** will trigger a warning but won't break generation

## Gender / Voice

| Tag | Description |
|-----|-------------|
| `male` | Male vocalist |
| `female` | Female vocalist |
| `male vocal` | Male vocalist |
| `female vocal` | Female vocalist |

## Genre

| Tag | Description |
|-----|-------------|
| `pop` | Pop music with catchy melodies |
| `rock` | Rock with electric guitars and drums |
| `hip-hop` | Hip-hop with rhythmic beats |
| `r&b` | R&B with soulful vocals |
| `electronic` | Electronic music |
| `dance` | Dance music |
| `edm` | Electronic dance music |
| `jazz` | Jazz with improvisation |
| `classical` | Classical orchestral |
| `country` | Country music |
| `folk` | Folk music |
| `metal` | Heavy metal |
| `funk` | Funky groove |
| `soul` | Soulful music |
| `reggae` | Reggae |
| `blues` | Blues |
| `indie` | Indie style |
| `indie pop` | Indie pop |
| `alternative` | Alternative rock |
| `ambient` | Ambient soundscape |
| `acoustic` | Acoustic instrumentation |
| `ballad` | Emotional ballad |
| `gospel` | Gospel music |
| `trance` | Trance music |
| `techno` | Techno |
| `house` | House music |
| `rap` | Rap |
| `trap` | Trap music |

### Regional Pop Styles

| Tag | Description |
|-----|-------------|
| `j-pop` | Japanese pop |
| `k-pop` | Korean pop |
| `c-pop` | Chinese pop |
| `cantopop` | Cantonese pop |
| `mandopop` | Mandarin pop |
| `anime` | Anime soundtrack style |

## Instruments

| Tag | Description |
|-----|-------------|
| `piano` | Featuring piano |
| `guitar` | Featuring guitar |
| `drums` | Prominent drums |
| `bass` | Prominent bass |
| `synthesizer` | Synthesizer sounds |
| `synth` | Synthesizer sounds |
| `strings` | String instruments |
| `keyboard` | Keyboard |
| `violin` | Featuring violin |
| `orchestra` | Orchestral arrangement |
| `pipa` | Featuring pipa |

## Mood / Emotion

| Tag | Description |
|-----|-------------|
| `happy` | Upbeat and cheerful |
| `sad` | Melancholic |
| `energetic` | High energy |
| `calm` | Peaceful |
| `relaxed` | Laid-back |
| `romantic` | Romantic |
| `aggressive` | Intense |
| `melancholic` | Melancholic |
| `dreamy` | Dreamy |
| `hopeful` | Hopeful |
| `cheerful` | Cheerful |
| `peaceful` | Peaceful |
| `warm` | Warm feeling |
| `joyful` | Joyful |
| `upbeat` | Upbeat |
| `emotional` | Emotional |
| `uplifting` | Uplifting |
| `chill` | Chill vibes |
| `nostalgic` | Nostalgic |
| `mellow` | Mellow |
| `playful` | Playful |
| `intense` | Intense |
| `epic` | Epic and grand |
| `passionate` | Passionate |
| `soft` | Soft |

## Tempo / Energy

| Tag | Description |
|-----|-------------|
| `fast` | Fast tempo |
| `slow` | Slow tempo |
| `heavy` | Heavy and powerful |
| `light` | Light and airy |
| `strong` | Strong and forceful |

## Singer Timbre

| Tag | Description |
|-----|-------------|
| `clear` | Clear vocals |
| `raspy` | Raspy voice |
| `smooth` | Smooth vocals |
| `powerful` | Powerful vocals |
| `breathy` | Breathy vocals |
| `deep` | Deep voice |
| `bright` | Bright vocal tone |
| `youthful` | Youthful voice |

## Scene / Context

| Tag | Description |
|-----|-------------|
| `wedding` | Wedding music |
| `workout` | Workout music |
| `meditation` | Meditation music |
| `study` | Study music |
| `party` | Party music |
| `club` | Club music |
| `cafe` | Cafe ambiance |
| `driving` | Driving music |
| `night` | Nighttime mood |
| `morning` | Morning mood |
| `sunset` | Sunset mood |
| `beach` | Beach vibes |
| `rainy day` | Rainy day mood |

## Example Tag Combinations

```
# Upbeat pop song
pop,happy,energetic,female vocal

# Chill R&B
r&b,chill,smooth,romantic

# Rock ballad
rock,ballad,emotional,guitar

# Electronic dance
electronic,dance,energetic,synthesizer

# Acoustic folk
folk,acoustic,calm,guitar

# Hip-hop
hip-hop,rap,bass,male vocal
```

## Known Limitations

1. **Genre diversity is limited** - The 3B model tends toward pop-style output regardless of tags
2. **Some tags are ignored** - Not all tags have strong conditioning effects
3. **Consistency varies** - The same tags may produce different results across generations

## Workaround: Style in Lyrics

If tags aren't working, try putting style descriptions directly in your lyrics:

```
[Style: This is an energetic rock song with electric guitars and powerful drums]

[Verse]
Your lyrics here...
```

The model often responds better to natural language descriptions in the lyrics field than to comma-separated tags.
