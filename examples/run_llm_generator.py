"""CLI example: generate lyrics and tags using the new LLM generator."""
from heartlib.llm.generator import get_generator
import argparse

parser = argparse.ArgumentParser(description="Generate lyrics + tags (local Transformers runtime)")
parser.add_argument("--style", default="electronic synthwave", help="Style or short instruction")
parser.add_argument("--length", default="medium", choices=["short","medium","long"], help="Target length")
parser.add_argument("--no-structure", dest="structure", action="store_false", help="Do not include structure tags")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--device", default="auto", help="Device to run on: auto, cuda, cpu, or cuda:0")
args = parser.parse_args()

g = get_generator(device=args.device)
out = g.generate_lyrics_and_tags(style=args.style, length=args.length, include_structure=args.structure, seed=args.seed)
print("=== LYRICS ===\n")
print(out['lyrics'])
print('\n=== TAGS ===\n')
print(out['tags'])
