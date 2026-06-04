from pathlib import Path
import os
import numpy as np
import torch


def local_run():
    if os.path.exists("/content"):
        # running in Google Colab
        return False
    else:
        # running locally
        return True


def get_base_dir():
    if local_run():
        # running locally
        return Path.cwd()
    else:
        # running in Google Colab
        return Path("/content/deforestation-unet")


# Function for setting the seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
