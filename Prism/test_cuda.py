import torch
print("CUDA Disponibile:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Dispositivo GPU:", torch.cuda.get_device_name(0))
