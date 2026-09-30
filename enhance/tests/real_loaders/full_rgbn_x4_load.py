import pathlib

import matplotlib.pyplot as plt
import safetensors.torch
from sen2sr.models.opensr_baseline.mamba import MambaSR
from sen2sr.models.tricks import HardConstraint
from sen2sr.nonreference import srmodel


# MLSTAC API -----------------------------------------------------------------------
def example_data(path: pathlib.Path, device: str = "cuda:0", *args, **kwargs):
    data_f = path / "example_data.safetensor"
    sample = safetensors.torch.load_file(data_f)
    return  sample["lr"].to(device), sample["hr"].to(device)

def trainable_model(path, device: str = "cuda:0", *args, **kwargs):
    trainable_f = path / "model.safetensor"

    # Load model parameters
    sr_model_weights = safetensors.torch.load_file(trainable_f)
    parameters = {
        "img_size": (128, 128),
        "in_channels": 4,
        "out_channels": 4,
        "embed_dim": 96,
        "depths": [8, 8, 8, 8, 8, 8],
        "num_heads": [8, 8, 8, 8, 8, 8],
        "mlp_ratio": 4,
        "upscale": 4,
        "attention_type": "sigmoid_02",
        "upsampler": "pixelshuffle",
        "resi_connection": "1conv",
        "operation_attention": "sum",
    }
    sr_model = MambaSR(**parameters)
    sr_model.load_state_dict(sr_model_weights)
    sr_model.to(device)

    # Load HardConstraint
    hard_constraint_weights = safetensors.torch.load_file(path / "hard_constraint.safetensor")
    hard_constraint = HardConstraint(low_pass_mask=hard_constraint_weights["weights"].to(device), device=device)

    return srmodel(sr_model, hard_constraint, device)


def compiled_model(path, device: str = "cuda:0", *args, **kwargs):
    trainable_f = path / "model.safetensor"

    # Load model parameters
    sr_model_weights = safetensors.torch.load_file(trainable_f)
    parameters = {
        "img_size": (128, 128),
        "in_channels": 4,
        "out_channels": 4,
        "embed_dim": 96,
        "depths": [8, 8, 8, 8, 8, 8],
        "num_heads": [8, 8, 8, 8, 8, 8],
        "mlp_ratio": 4,
        "upscale": 4,
        "attention_type": "sigmoid_02",
        "upsampler": "pixelshuffle",
        "resi_connection": "1conv",
        "operation_attention": "sum",
    }
    sr_model = MambaSR(**parameters)
    sr_model.load_state_dict(sr_model_weights)
    sr_model = sr_model.eval()
    sr_model.to(device)
    for param in sr_model.parameters():
        param.requires_grad = False

    # Load HardConstraint
    hard_constraint_weights = safetensors.torch.load_file(path / "hard_constraint.safetensor")
    hard_constraint = HardConstraint(low_pass_mask=hard_constraint_weights["weights"].to(device), device=device)

    return srmodel(sr_model, hard_constraint, device)


def display_results(path: pathlib.Path, device: str = "cuda:0", *args, **kwargs):
    # Load model
    model = compiled_model(path, device)

    # Load data
    lr, hr = example_data(path, device)

    # Run model
    SuperX = model(lr)

    #Display results
    Xrgb = lr[0, 0:3].cpu().numpy().transpose(1, 2, 0)
    SuperXrgb = SuperX[0, 0:3].cpu().numpy().transpose(1, 2, 0)
    lr_slice = slice(16, 32+80)
    hr_slice = slice(lr_slice.start*4, lr_slice.stop*4)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(Xrgb[lr_slice, lr_slice]*3)
    ax[0].set_title("Sentinel-2")
    ax[1].imshow(SuperXrgb[hr_slice, hr_slice]*3)
    ax[1].set_title("Super-Resolved")
    ax[2].imshow(hr[0, 0:3].cpu().numpy().transpose(1, 2, 0)[hr_slice, hr_slice]*3)
    ax[2].set_title("True HR")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    return fig
