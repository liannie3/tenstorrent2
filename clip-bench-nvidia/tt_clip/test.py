import device

device.open_ttnn()
dev = device.get_device()
print(dev)
device.close_ttnn()