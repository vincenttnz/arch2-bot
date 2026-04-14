import argparse

import torch

from sensenova_si import SenseNovaSIBagelModel


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="BAGEL image generation example - generate image from text prompt"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="sensenova/SenseNova-SI-1.1-BAGEL-7B-MoT",
        help="BAGEL model path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A chubby cat made of 3D point clouds, stretching its body, translucent with a soft glow.",
        help="Text prompt used to generate an image",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "think_generate"],
        help="BAGEL mode: generate or think_generate",
    )
    parser.add_argument(
        "--out_img_dir",
        type=str,
        default="./output_images/test_bagel/",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16"],
        help="Model precision type",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed()

    print(f"Model path: {args.model_path}")
    print(f"Mode: {args.mode}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)

    # Initialize BAGEL model with generate mode
    print("Loading model...")
    model = SenseNovaSIBagelModel(
        model_path=args.model_path,
        mode=args.mode,
        out_img_dir=args.out_img_dir,
        dtype=args.dtype,
    )

    print("Generating image...")
    # Call generate with the prompt; images not needed for generate mode
    generated_image_path = model.generate(question=args.prompt, images=None)

    print("-" * 50)
    print("Done!")
    print(f"Image saved to: {generated_image_path}")


if __name__ == "__main__":
    main()
