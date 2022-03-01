from math import sqrt

n = int(input())
k= 0
for i in range(1, int(sqrt(n) + 1)):
    if n % i == 0 and i % 2 == n // i % 2:
        k+= 1
print(k)
