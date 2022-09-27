
# Minimize the function f(x) = 2*x^2
# Gradient descent starting from x = 5

x = 5
alpha = 0.01
slope = 4*x

while slope > 10**-10:
    slope = 4*x
    x = x - alpha*slope

f = 2*x**2

print(f"x={x}\tf(x)={f}")
# x=2.3457834429805036e-11	f(x)=1.1005399922722931e-21
#  = basically zero             = basically zero
