from heartlib import HeartMuLaGenPipeline
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--version", type=str, default="3B")
    parser.add_argument("--lyrics", type=str, default="./assets/lyrics.txt")
    parser.add_argument("--tags", type=str, default="./assets/tags.txt")
    parser.add_argument("--save_path", type=str, default="./assets/output.mp3")

    parser.add_argument("--max_audio_length_ms", type=int, default=240_000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    
    # New: lazy loading toggle
    parser.add_argument("--lazy_load", action="store_true", default=True,
                       help="Enable lazy loading to load model components on-demand and save GPU memory")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Display initial GPU memory usage
    print(f"Initial GPU memory usage: {torch.cuda.memory_allocated('cuda') / 1024**3:.2f} GB")
    
    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        version=args.version,
        lazy_load=args.lazy_load,
    )
    
    print(f"GPU memory usage after pipeline initialization: {torch.cuda.memory_allocated('cuda') / 1024**3:.2f} GB")
    
    with torch.no_grad():
        pipe(
            {
                "lyrics": args.lyrics,
                "tags": args.tags,
            },
            max_audio_length_ms=args.max_audio_length_ms,
            save_path=args.save_path,
            topk=args.topk,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )
    
    print(f"GPU memory usage after completion: {torch.cuda.memory_allocated('cuda') / 1024**3:.2f} GB")
    print(f"Generated music saved to {args.save_path}")