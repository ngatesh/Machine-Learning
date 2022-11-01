import torch as t

x1 = t.tensor([-1.0], requires_grad=True).float()
x2 = t.tensor([2.0], requires_grad=True).float()
x3 = t.tensor([3.0], requires_grad=True).float()


def f(a, b, c):
    return a*b*t.sin(b*c)


F = f(x1, x2, x3)
F.backward()

print("Autograd")
print(f"df/dx1: {x1.grad}")
print(f"df/dx2: {x2.grad}")
print(f"df/dx3: {x3.grad}")

h = t.tensor([0.001], requires_grad=False).float()

dx1 = (f(x1 + h, x2, x3) - f(x1 - h, x2, x3)) / (2*h)
dx2 = (f(x1, x2 + h, x3) - f(x1, x2 - h, x3)) / (2*h)
dx3 = (f(x1, x2, x3 + h) - f(x1, x2, x3 - h)) / (2*h)

print("\nCenter Finite Difference")
print(f"df/dx1: {dx1}")
print(f"df/dx2: {dx2}")
print(f"df/dx3: {dx3}")
