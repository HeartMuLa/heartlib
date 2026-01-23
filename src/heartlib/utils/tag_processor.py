"""
Tag processing utilities for HeartMuLa.

This module provides utilities to improve tag conditioning by:
1. Converting comma-separated tags to natural language descriptions
2. Validating tags against known working tags
3. Enhancing lyrics with style descriptions for better conditioning

The 3B model responds better to natural language style descriptions in the lyrics
than to comma-separated tags. This module exploits that behavior.
"""

from typing import List, Set, Tuple

# Tag-to-description mappings for natural language injection
TAG_TO_DESCRIPTION = {
    # Genre mappings
    "pop": "pop music with catchy melodies",
    "rock": "rock music with electric guitars and drums",
    "hip-hop": "hip-hop with rhythmic beats and flow",
    "hip hop": "hip-hop with rhythmic beats and flow",
    "r&b": "R&B with soulful vocals and smooth grooves",
    "rnb": "R&B with soulful vocals and smooth grooves",
    "electronic": "electronic music with synthesizers",
    "dance": "dance music with driving beats",
    "edm": "electronic dance music",
    "jazz": "jazz with improvisation and swing",
    "classical": "classical orchestral arrangement",
    "country": "country music with acoustic guitar",
    "folk": "folk music with acoustic instruments",
    "metal": "heavy metal with distorted guitars",
    "funk": "funky groove with bass and rhythm",
    "soul": "soulful music with emotional vocals",
    "reggae": "reggae with offbeat rhythms",
    "blues": "blues with expressive guitar",
    "indie": "indie style",
    "indie pop": "indie pop style",
    "alternative": "alternative rock style",
    "ambient": "ambient atmospheric soundscape",
    "acoustic": "acoustic instrumentation",
    "ballad": "emotional ballad",
    "gospel": "gospel music",
    "christian": "christian music",
    "anime": "anime soundtrack style",
    "j-pop": "Japanese pop style",
    "k-pop": "Korean pop style",
    "c-pop": "Chinese pop style",
    "cantopop": "Cantonese pop style",
    "mandopop": "Mandarin pop style",
    "trance": "trance music with hypnotic beats",
    "techno": "techno with driving electronic beats",
    "house": "house music with four-on-the-floor beat",
    "rap": "rap with rhythmic vocal delivery",
    "trap": "trap music with heavy bass",
    
    # Mood mappings
    "happy": "upbeat and cheerful",
    "sad": "melancholic and emotional",
    "energetic": "high energy and exciting",
    "calm": "peaceful and relaxing",
    "relaxed": "laid-back and chill",
    "romantic": "romantic and intimate",
    "aggressive": "intense and powerful",
    "melancholic": "melancholic mood",
    "dreamy": "dreamy and ethereal",
    "hopeful": "hopeful and uplifting",
    "sentimental": "sentimental feeling",
    "gentle": "gentle and soft",
    "cheerful": "cheerful and bright",
    "peaceful": "peaceful atmosphere",
    "warm": "warm feeling",
    "sweet": "sweet and tender",
    "joyful": "joyful and celebratory",
    "solemn": "solemn and serious",
    "upbeat": "upbeat tempo",
    "reflective": "reflective and thoughtful",
    "emotional": "deeply emotional",
    "confident": "confident and bold",
    "uplifting": "uplifting and inspiring",
    "chill": "chill vibes",
    "nostalgic": "nostalgic feeling",
    "introspective": "introspective mood",
    "mellow": "mellow and smooth",
    "playful": "playful and fun",
    "intense": "intense and dramatic",
    "rebellious": "rebellious attitude",
    "epic": "epic and grand",
    "passionate": "passionate expression",
    "soft": "soft and delicate",
    "gloomy": "dark and gloomy",
    
    # Tempo/energy mappings
    "fast": "fast tempo",
    "slow": "slow tempo",
    "fast tempo": "fast tempo",
    "slow tempo": "slow tempo",
    "heavy": "heavy and powerful",
    "light": "light and airy",
    "strong": "strong and forceful",
    
    # Instrument mappings
    "piano": "featuring piano",
    "guitar": "featuring guitar",
    "drums": "with prominent drums",
    "synthesizer": "with synthesizer sounds",
    "synth": "with synthesizer sounds",
    "strings": "with string instruments",
    "bass": "with prominent bass",
    "keyboard": "with keyboard",
    "violin": "featuring violin",
    "orchestra": "with orchestral arrangement",
    "pipa": "featuring pipa",
    
    # Voice mappings
    "male vocal": "sung by a male vocalist",
    "female vocal": "sung by a female vocalist",
    "male voice": "sung by a male vocalist",
    "female voice": "sung by a female vocalist",
    "male": "male vocalist",
    "female": "female vocalist",
    
    # Singer timbre
    "clear": "clear vocals",
    "raspy": "raspy voice",
    "smooth": "smooth vocals",
    "powerful": "powerful vocals",
    "breathy": "breathy vocals",
    "deep": "deep voice",
    "bright": "bright vocal tone",
    "youthful": "youthful voice",
    "mature": "mature voice",
}

# Set of all known tags that have shown some effect
KNOWN_TAGS: Set[str] = set(TAG_TO_DESCRIPTION.keys())

# Additional tags from community testing (Issue #9)
KNOWN_TAGS.update({
    # Scenes
    "driving", "cafe", "relaxing", "meditation", "night", "alone", "walking",
    "travel", "study", "wedding", "workout", "dating", "bedroom", "home",
    "rainy day", "evening", "club", "reflection", "church", "sunset", "gaming",
    "reading", "morning", "street", "thinking", "shopping", "worship",
    "campfire", "movie", "playground", "funeral", "party", "beach",
    
    # Topics
    "love", "relationship", "longing", "loss", "regret", "life", "hope",
    "breakup", "heartbreak", "memory", "farewell", "loneliness", "youth",
    "nature", "romance", "dream", "faith", "friendship", "lost", "rebellion",
    "encouragement", "freedom", "struggle", "self-reflection", "missing",
    "inspiration", "happiness", "courage", "reminiscence", "confess",
})


def tags_to_description(tags: str) -> str:
    """
    Convert comma-separated tags to a natural language description.
    
    The model responds better to natural language descriptions than to
    comma-separated tags. This function converts tags like "rock,energetic"
    into "[Style: rock music with electric guitars and drums, high energy and exciting]"
    
    Args:
        tags: Comma-separated tag string (e.g., "rock,energetic,guitar")
        
    Returns:
        Natural language description string, or empty string if no tags
    """
    if not tags or not tags.strip():
        return ""
    
    tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
    if not tag_list:
        return ""
    
    descriptions = []
    for tag in tag_list:
        if tag in TAG_TO_DESCRIPTION:
            descriptions.append(TAG_TO_DESCRIPTION[tag])
        else:
            # Unknown tag - include as-is with "style" suffix
            descriptions.append(f"{tag} style")
    
    if descriptions:
        return "[Style: " + ", ".join(descriptions) + "]\n\n"
    return ""


def enhance_lyrics_with_tags(lyrics: str, tags: str) -> str:
    """
    Prepend tag descriptions to lyrics for better style conditioning.
    
    This exploits the model's tendency to extract style information from
    the lyrics field rather than the tags field.
    
    Args:
        lyrics: The original lyrics text
        tags: Comma-separated tag string
        
    Returns:
        Lyrics with style description prepended
    """
    description = tags_to_description(tags)
    return description + lyrics


def validate_tags(tags: str) -> Tuple[List[str], List[str]]:
    """
    Validate tags against known working tags.
    
    Args:
        tags: Comma-separated tag string
        
    Returns:
        Tuple of (known_tags, unknown_tags) lists
    """
    if not tags or not tags.strip():
        return [], []
    
    tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
    
    known = []
    unknown = []
    
    for tag in tag_list:
        if tag in KNOWN_TAGS:
            known.append(tag)
        else:
            unknown.append(tag)
    
    return known, unknown


def get_tag_warnings(tags: str) -> List[str]:
    """
    Get warning messages for unknown tags.
    
    Args:
        tags: Comma-separated tag string
        
    Returns:
        List of warning messages
    """
    _, unknown = validate_tags(tags)
    warnings = []
    
    for tag in unknown:
        warnings.append(
            f"Warning: Tag '{tag}' is not in the known working tags list. "
            "It may not affect the output. See SUPPORTED_TAGS.md for guidance."
        )
    
    return warnings
