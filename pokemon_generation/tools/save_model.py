import os
import torch


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_dir: str,
    filename: str = "best_model.pth",
) -> None:
    """
    Saves the model state, optimizer state, current epoch, and loss to a specified directory.

    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state will be saved.
        epoch (int): The current epoch number (used for checkpointing).
        loss (float): The current loss value to track model performance.
        save_dir (str): The dictionay where the model checkpoint will be saved.
        filename (str, optional): The name of the file to save the model. Defaults to "best_model.pth".

    Returns:
        None
    """
    save_path = os.path.join(save_dir, filename)

    # Save the model, optimizer state, and additional metadata (epoch and loss)
    torch.save(
        {
            "epoch": epoch + 1,  # Save epoch + 1 for easier tracking
            "model_state_dict": model.state_dict(),  # Save model weights
            "optimizer_state_dict": optimizer.state_dict(),  # Save optimizer state (important for resuming training)
            "loss": loss,  # Save the current loss value
        },
        save_path,
    )

    # Print a confirmation message indicating the model has been saved.
    print(f"Mode saved at {save_path} (Loss: {loss: .4f}, Epoch: {epoch + 1})")
