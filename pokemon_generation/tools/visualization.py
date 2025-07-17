# visualization.py

from typing import List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def pixel_to_image(pixel_color: List[int], colormap: List[List[int]]):
    """
    Converts a list of pixel indices into a 20x20 RGB image using a colormap.

    Args:
        pixel_color (List[int]): A list of pixel indices representing colors.
        colormap (List[List[int]]): A list where each index maps to an RGB color [R, G, B].

    Returns:
        Image.Image: A PIL Image object representing the reconstructed image.
    """
    # Ensure pixel_color list has at least 400(20x20) elements (pad whth 0s).
    if len(pixel_color) < 400:
        pixel_color = pixel_color + [0] * (400 - len(pixel_color))

    # Map pixel indices to RGB colors by colormap
    pixel_data = [colormap[pixel] for pixel in pixel_color]

    # Convert to numpy array and reshape to 20x20x3 (RGB iamge)
    image_array = np.array(pixel_data, dtype=np.uint8).reshape(20, 20, 3)

    # Create a PIL image from the array
    image = Image.fromarray(image_array)

    return image


def show_images(images: List[Image.Image]) -> None:
    """
    Display a grid of up to 96 images using Matplotlib.

    Args:
        images (List[Image.Image]): A list of PIL Image object to display.

    Returns:
        None
    """
    # Limit to 96 images to show
    n_images = min(96, len(images))
    # Set up the figure size and grid layout (6 rows, 16 columns)
    fig, axes = plt.subplots(6, 16, figsize=(16, 6))

    # Flatten to make iteration easier
    axes = axes.flatten()
    # Loop through images and display each one in the grid
    for i, ax in enumerate(axes):
        if i < n_images:
            ax.imshow(images[i])
            ax.axis("off")  # Hide axis
        else:
            ax.axis("off")  # Hide unused subplots

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
