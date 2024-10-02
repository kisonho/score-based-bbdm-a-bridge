import argparse, deeplabv3, torch


class Configs:
    model_state_dict_path: str
    output_path: str
    num_classes: int


if __name__ == "__main__":
    # initialize argument parser
    parser = argparse.ArgumentParser(description="DeepLabV3+ Converter to convert the pre-trained state dictionary.")
    parser.add_argument("model_state_dict_path", type=str, help="The path of saved model state dict to convert.")
    parser.add_argument("output_path", type=str, help="The output path of converted full model.")
    parser.add_argument("-n", "--num_classes", type=int, default=19, help="The number of classes, default is 19.")

    # get arguments
    args = parser.parse_args(namespace=Configs())

    # load model state dict
    ckpt_state_dict: dict[str, torch.Tensor] = torch.load(args.model_state_dict_path, map_location='cpu')['model_state']
    ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if 'loss' not in k}
    model = deeplabv3.deeplabv3_resnet101(num_classes=args.num_classes, pretrained_backbone=False)
    model_state_dict: dict[str, torch.Tensor] = model.state_dict()
    model.load_state_dict(ckpt_state_dict)

    # save model
    torch.save(model, args.output_path)
