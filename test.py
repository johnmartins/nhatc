
X = [1, 2, 3, 4, 5]
Y = [5, 4, 3, 2, 1]
Z = [9, 9, 9, 9, 9]

problems = [lambda X, Y, Z: (X[0] + Y[2]) / 2]

res = problems[0](X, Y, Z)
print(res)
