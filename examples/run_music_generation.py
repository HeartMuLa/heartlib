import argparse

import torch

from heartlib import HeartMuLaGenPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--version", type=str, default="3B")
    parser.add_argument("--lyrics", type=str, default="./assets/lyrics.txt")
    parser.add_argument("--tags", type=str, default="./assets/tags.txt")
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Optional: path to reference audio for MuQ-MuLan conditioning.",
    )
    parser.add_argument(
        "--load_muq_mulan",
        action="store_true",
        help="Auto-download/load MuQ-MuLan from Hugging Face (requires `pip install muq`).",
    )
    parser.add_argument(
        "--muq_model_id",
        type=str,
        default="OpenMuQ/MuQ-MuLan-large",
        help="Hugging Face model id for MuQ-MuLan.",
    )
    parser.add_argument(
        "--muq_cache_dir",
        type=str,
        default=None,
        help="Optional: Hugging Face cache dir for MuQ-MuLan.",
    )
    parser.add_argument(
        "--muq_revision",
        type=str,
        default=None,
        help="Optional: Hugging Face revision (branch/tag/commit) for MuQ-MuLan.",
    )
    parser.add_argument(
        "--muq_segment_sec",
        type=float,
        default=10.0,
        help="Reference-audio segment length (seconds) fed to MuQ.",
    )
    parser.add_argument(
        "--muq_sample_rate",
        type=int,
        default=24000,
        help="Sample rate expected by MuQ (usually 24 kHz).",
    )
    parser.add_argument("--save_path", type=str, default="./assets/output.mp3")
    parser.add_argument(
        "--codes_path",
        type=str,
        default=None,
        help="Optional: save generated audio token frames (torch .pt) for analysis.",
    )

    parser.add_argument("--max_audio_length_ms", type=int, default=240_000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # MPS commonly lacks bf16 support; fp16 is the safest default.
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.bfloat16

    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device=device,
        dtype=dtype,
        version=args.version,
        load_muq_mulan=args.load_muq_mulan,
        muq_model_id=args.muq_model_id,
        muq_cache_dir=args.muq_cache_dir,
        muq_revision=args.muq_revision,
    )
    with torch.no_grad():
        pipe(
            {
                "lyrics": args.lyrics,
                "tags": args.tags,
                "ref_audio": args.ref_audio,
                "muq_segment_sec": args.muq_segment_sec,
                "muq_sample_rate": args.muq_sample_rate,
            },
            max_audio_length_ms=args.max_audio_length_ms,
            save_path=args.save_path,
            codes_path=args.codes_path,
            topk=args.topk,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )
    print(f"Generated music saved to {args.save_path}")
    if args.codes_path:
        print(f"Saved audio token frames to {args.codes_path}")
