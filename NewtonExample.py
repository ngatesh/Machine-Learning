
"""
 Solve:
  x = sqrt(2)
  f(x) = x^2 - 2 = 0
"""

f = 0
slope = 0

guess = 1
guess_1 = 1

r = 1

while abs(r) > 0.000001:
    f = pow(guess, 2) - 2
    slope = 2*guess

    guess_1 = guess
    guess = guess - f / slope

    r = guess - guess_1
    print(guess)

print(pow(2, 0.5))


