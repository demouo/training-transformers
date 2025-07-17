import os
from typing import List
from datasets import load_dataset
from tqdm import tqdm
from pokemon_generation.tools.data_helper import PixelSequenceDataset
from pokemon_generation.tools.visualization import pixel_to_image, show_images
from torch.utils.data import DataLoader
from transformers import GPT2Config, AutoModelForCausalLM
import torch
from torch import nn
import torch.optim as optim
from pokemon_generation.tools.save_model import save_model


def main(mode: str = "train") -> None:
    """
    Using Transformers to train the pokemon image generation, token by token.

    Args:
        mode (str): The running mode, either 'train' or 'test'. Defaults to 'train'.
            - 'train': Train and Test the model
            - 'test': Only Test the model performance

    Returns:
        None
    """
    # Load the pokemon/colormap dataset from local downloaded content
    pokemon_dataset = load_dataset("data/pokemon")
    colormap = list(load_dataset("data/colormap")["train"]["color"])

    # Define number of classes that will be learned
    n_classes = len(colormap)

    # Define batch size
    batch_size = 16

    # ----- Prepare Dataset and DataLoader for Training ----
    train_dataset: PixelSequenceDataset = PixelSequenceDataset(
        pokemon_dataset["train"]["pixel_color"], "train"
    )
    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # ----- Prepare Dataset and DataLoader for Validation ----
    dev_dataset: PixelSequenceDataset = PixelSequenceDataset(
        pokemon_dataset["validation"]["pixel_color"], "dev"
    )
    dev_dataloader: DataLoader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False
    )

    # ----- Prepare Dataset and DataLoader for Testing ----
    test_dataset: PixelSequenceDataset = PixelSequenceDataset(
        pokemon_dataset["test"]["pixel_color"], "test"
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    ## show images from datasets
    # train_images = [
    #     pixel_to_image(pixel_color=data["pixel_color"], colormap=colormap)
    #     for data in pokemon_dataset["train"]
    # ]
    # show_images(train_images)

    # Load Model from config
    gpt2_config = {
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 128,
        "n_embd": 64,
        "n_head": 2,
        "n_layer": 2,
        "n_positions": 400,
        "resid_pdrop": 0.1,
        "vocab_size": n_classes,
        "pad_token_id": None,
        "eos_token_id": None,
    }
    # Load GPT-2 model configuration from dictionary
    config = GPT2Config.from_dict(gpt2_config)
    # Load the model by the configuration
    model = AutoModelForCausalLM.from_config(config)
    # Show the model detailed info
    print(model)
    # Calculate the trainable parameters
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_parameters:,}")

    # ---- train ----
    # Training hyperparameters
    epoches = 50  # Number of training epochs
    learning_rate = 1e-03  # Learning rate for optimizer

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps"
    )  # Check if CUDA is available for GPU
    save_dir = "checkpoints"  # Directory to save the checkpoint

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Loss function for classification tasks
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.1
    )  # AdamW optimizer with weight decay

    # Ensure the saving directory exist.
    os.makedirs(save_dir, exist_ok=True)
    # Initialize the best loss as positive infinity for comparision during model checkpointing.
    best_loss: float = float("inf")
    # Move model to the appropriate device (GPU or CPU or MPS)
    model.to(device)

    # Traing loop
    if mode == "train":
        for epoch in range(epoches):
            model.train()  # Set the model to training mode
            epoch_loss = 0.0  # Initialize the epoch loss

            # Iterate over training data batches
            for input_ids, labels in tqdm(
                train_dataloader, desc=f"Training Epoch: {epoch}"
            ):
                # Move data to the same device as the model
                input_ids, labels = input_ids.to(device), labels.to(device)
                # [batch_size, sequence], [batch_size, sequence]

                # --- TODO why writing like this? ---
                # See the shape of them.

                # Forward pass through the model to get logits (output probabilities)
                outputs = model(input_ids=input_ids).logits.view(-1, config.vocab_size)
                # [batch_size, sequence, vocab_size] -> [batch_size * sequence, vocab_size]

                # Flatten labels to match logits shape
                labels = labels.view(-1)
                # [batch_size, sequence] -> [batch_size * sequence]

                # Calculate loss by CrossEntropyLoss
                # It needs, [N, C], [N], Where the last one (labels) is the true idx
                # to select it's propability in the previous one (outputs) for calculating loss.
                loss = criterion(outputs, labels)

                # Backpropagation and optimizer step
                optimizer.zero_grad()  # ! Be careful to reset the gradients to zero
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model weights

                # Accumulate the loss for the epoch
                epoch_loss += loss.item()

            # Compute average epoch loss
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epoches}, Loss: {avg_epoch_loss: .4f}")

            # Evaluation Loop(Validation)
            model.eval()  # Set the model to evaluation mode (disable dropout, etc.)
            total_accuracy = 0  # Initialize total accuracy
            n_batches = 0  # Initialize batch counter

            with torch.no_grad():  # Disable gradient calculation for validation
                # Iterate through validation data batches
                for input_ids, labels in tqdm(dev_dataloader, desc="Evaluating"):
                    input_ids, labels = input_ids.to(device), labels.to(device)
                    attention_mask = torch.ones_like(input_ids)

                    # ---- TODO Why use attention mask? ----
                    # Attention mask tells the self-attention calculator that
                    # which token needs to be involved with 1, ignored with 0 otherwise.
                    # Here is all input ids need to be involed without any padding.

                    # Perform batch inference using the model
                    generated_outputs = model.generate(
                        input_ids, attention_mask=attention_mask, max_length=400
                    )

                    # Extract the last 160 token from generated outputs and labels
                    generated_outputs = generated_outputs[:, -160:]

                    # Calculate accuracy for the batch
                    accuracy = (generated_outputs == labels).float().mean().item()
                    total_accuracy += accuracy
                    n_batches += 1

            # Compute average reconstruction accuracy for the epoch
            avg_accuracy = total_accuracy / n_batches
            print(
                f"Epoch: {epoch + 1}/{epoches}, Reconstruction Accuracy: {avg_accuracy:.4f}"
            )

            # Save the model state if current epoch loss is better (lower) than previous best loss
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_model(model, optimizer, epoch, best_loss, save_dir)

    # Inference
    filename = "best_model.pth"
    # Load the best model from saved checkpoint
    best_model_path = os.path.join(save_dir, filename)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"The model path {best_model_path} not found.")
    checkpoint = torch.load(best_model_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode (disable dropout, etc.)

    # Testing Loop with Batch Inference
    # Store the generated sequences from the model
    results: List = []

    with torch.no_grad():  # Disable gradients caculation for inference
        # Iterate through test data in batches
        for input_ids in tqdm(test_dataloader, desc="Generating Outputs"):
            # Move data to the same device as model
            input_ids = input_ids.to(device)
            # Use Attention mask to ensure valid token positions
            attention_mask = torch.ones_like(input_ids)

            # Generate predictions for the entire batch
            generated_outputs = model.generate(
                input_ids, attention_mask=attention_mask, max_length=400
            )
            # Convert batch outputs to a list and append to results
            batch_results = generated_outputs.cpu().numpy().tolist()
            # Extend results list with batch results
            results.extend(batch_results)

    # Save the results to a file
    output_file: str = "reconstructed_results.txt"
    with open(output_file, "w") as fp:
        for sequence in results:
            fp.write(" ".join(map(str, sequence)) + "\n")

    # Show the generated results
    test_images = [
        pixel_to_image(pixel_color=pixel_color, colormap=colormap)
        for pixel_color in results
    ]
    show_images(test_images)


if __name__ == "__main__":
    main(mode="test")
