import argparse
import os
from PIL import Image
import string
import torch
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--image-dir', help='Directory containing images to read')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cased', action='store_true', help='Cased comparison')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    charset_test = string.digits + string.ascii_lowercase
    if args.cased:
        charset_test += string.ascii_uppercase

    kwargs.update({'charset_test': charset_test})
    print(f'Additional keyword arguments: {kwargs}')

    if not args.image_dir:
        raise ValueError("Image directory must be specified with --image-dir")

    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    for fname in os.listdir(args.image_dir):
        full_path = os.path.join(args.image_dir, fname)
        if os.path.isfile(full_path) and fname.lower().endswith(('png', 'jpg', 'jpeg')):
            image = Image.open(full_path).convert('RGB')
            image = img_transform(image).unsqueeze(0).to(args.device)

            p = model(image).softmax(-1)
            pred, p = model.tokenizer.decode(p)
            print(f'{fname}: {pred[0]}')

if __name__ == '__main__':
    main()