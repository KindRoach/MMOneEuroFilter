import numpy as np

from OneEuroFilterMD import OneEuroFilterMD

frames = 100
start = 0
end = 4 * np.pi
scale = 0.05

# The noisy signal
t = np.linspace(start, end, frames)
x_s = np.sin(t)
x_c = np.cos(t)
x_noisy = x_s + np.random.normal(scale=scale, size=len(t))
y_noisy = x_c + np.random.normal(scale=scale, size=len(t))
x = np.stack((x_noisy, y_noisy))

x_filtered = []
f = OneEuroFilterMD(1, 0.05, 1)
for i in range(len(t)):
    x_filtered.append(f(t[i], x[:, i]))

x_filtered = np.stack(x_filtered).T

print(x)
print(x_filtered)
