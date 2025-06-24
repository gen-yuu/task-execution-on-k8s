import torch

from common.utils import get_system_info

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    system_info = get_system_info(device)
    print("System Information:", system_info)
