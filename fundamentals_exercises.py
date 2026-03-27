import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

RANDOM_SEED: int = 7 #*0 for the 1st block and 1234 for the 2nd block
torch.cuda.manual_seed(RANDOM_SEED)
random1: torch.Tensor = torch.rand(7, 7)
random2: torch.Tensor = torch.rand(1, 7)
mult: torch.Tensor = torch.matmul(random1, random2.T)
# print(random1)
# print(random2)
print(mult)

torch.manual_seed(RANDOM_SEED)
random1 = torch.rand([2, 3], device="cpu").cuda()
random2 = torch.rand([2, 3], device="cpu").cuda()
mult = random1 @ random2.T
print(mult)
print(f"Minimum at position {mult.argmin()} is {torch.min(mult)}")
print(f"Maximum at position {torch.argmax(mult)} is {mult.max()}")

random1 = torch.randint(low=0, high=10, size=[1, 1, 1, 10])
squeezed: torch.Tensor = torch.squeeze(random1)
print(random1, random1.shape)
print(squeezed, squeezed.shape)