import torch

def normalization(image: torch.Tensor):
    """
    Normalize images to [0.0, 1.0].

    Arguments:
        image (torch.Tensor): Input image tensor
    
    Returns:
        image
    """

    if image.max() > 1.0:
        image /= 255.0
    return image


def reshape_image(image: torch.Tensor, target: tuple):
    """
    Reshpae image to the shape of the target.

    Arguments:
        image (torch.Tensor): Input image tensor
        target_shape (tuple): Output image shape
    
    Returns:
        image.view
    """

    return image.view(*target)