import argparse, torch, vqgan


class Configs:
    assert_weights_mapping: bool
    model_state_dict_path: str
    output_path: str
    in_channels: int


if __name__ == "__main__":
    # initialize argument parser
    parser = argparse.ArgumentParser(description="VQGAN Converter to convert the official VQGAN state dictionary.")
    parser.add_argument("model_state_dict_path", type=str, help="The path of saved model state dict to convert.")
    parser.add_argument("output_path", type=str, help="The output path of converted full model.")
    parser.add_argument("-c", "--in_channels", type=int, default=3, help="The number of input channels, default is 3.")
    parser.add_argument("--assert_weights_mapping", action="store_true", help="Assert the weights mapping between the official VQGAN and this implementation.")

    # get arguments
    args = parser.parse_args(namespace=Configs())

    # load model state dict
    ckpt_state_dict: dict[str, torch.Tensor] = torch.load(args.model_state_dict_path, map_location='cpu')
    ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() if 'loss' not in k}
    model = vqgan.build(args.in_channels, f=4)

    # get model state dict
    full_state_dict = model.state_dict()
    model_state_dict = {k: v for k, v in full_state_dict.items() if 'discriminator' not in k}

    # convert state dict
    saved_weights_names: list[str] = [n for n in ckpt_state_dict.keys()]
    saved_weights_values: list[torch.Tensor] = [t for t in ckpt_state_dict.values()]

    # loop weights
    for i, (name, t) in enumerate(model_state_dict.items()):
        if i < len(saved_weights_values):
            print(saved_weights_names[i], "->", name)
            t.copy_(saved_weights_values[i])
        elif args.assert_weights_mapping:
            raise RuntimeError(f"Cannot find the weight mapping for {name}.")
        else:
            print("None", "->", name)

    # check overflow weights
    if len(saved_weights_names) > len(model_state_dict) and args.assert_weights_mapping:
        raise RuntimeError(f"Overflow weights: {saved_weights_names[len(model_state_dict):]}")
    elif len(saved_weights_names) > len(model_state_dict):
        for i in range(len(model_state_dict), len(saved_weights_names)):
            print(saved_weights_names[i], "->", "None")

    # save full state dict
    full_state_dict.update(model_state_dict)
    model.load_state_dict(full_state_dict)

    # save model
    torch.save(model, args.output_path)
