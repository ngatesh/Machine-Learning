
mal = 0.50
fem = 0.50
D_mal = 0.0025
D_fem = 0.05

# P(mal|D) = P(D|mal)P(mal)/P(D)
#          = P(D|mal)P(mal)/(P(D|mal)P(mal)+P(D|fem)P(fem))

P_mal_D = D_mal*mal/(D_mal*mal + D_fem*fem)

print(f"Probability that a person is male, given that they have the Dercum Disease: {P_mal_D}")
