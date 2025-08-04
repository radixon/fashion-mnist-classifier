from datetime import datetime
from typing import List

def get_timestamp_str() -> str:
    """
    Generates a timestamp string in YYYYMMDD_HHMMSS format.
    """

    return datetime.now().strftime("%Y%m%d_%H%M%S")

FASHION_MNIST_CLASSES = ['T-Shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
