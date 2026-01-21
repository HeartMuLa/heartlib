from heartlib import HeartMuLaGenPipeline
import argparse
import torch


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "y", "true", "t", "1"):
        return True
    elif value.lower() in ("no", "n", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected. Got: {value}")


def str2dtype(value):
    value = value.lower()
    if value == "float32" or value == "fp32":
        return torch.float32
    elif value == "float16" or value == "fp16":
        return torch.float16
    elif value == "bfloat16" or value == "bf16":
        return torch.bfloat16
    else:
        raise argparse.ArgumentTypeError(f"Dtype not recognized: {value}")


def str2device(value):
    value = value.lower()
    return torch.device(value)


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
    parser.add_argument("--mula_device", type=str2device, default="cuda")
    parser.add_argument("--codec_device", type=str2device, default="cuda")
    parser.add_argument("--mula_dtype", type=str2dtype, default="bfloat16")
    parser.add_argument("--codec_dtype", type=str2dtype, default="float32")
    parser.add_argument("--lazy_load", type=str2bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device={
            "mula": torch.device(args.mula_device),
            "codec": torch.device(args.codec_device),
        },
        dtype={
            "mula": args.mula_dtype,
            "codec": args.codec_dtype,
        },
        version=args.version,
        lazy_load=args.lazy_load,
    )
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
    print(f"Generated music saved to {args.save_path}")
