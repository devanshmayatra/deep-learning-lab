import torch


def save_checkpoint(path, model, optimizer=None, epoch=0):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    
def load_checkpoint(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    if optimizer is not None and "optimizer_state" not in checkpoint:
        raise ValueError("This checkpoint has no optimizer state.")

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint["epoch"]