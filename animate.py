
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
 
fig = plt.figure(figsize=(6, 3))
x = [0]
y = [0]
 
ln, = plt.plot(x, y, '-')
plt.axis([0, 100, 0, 10])
 
def update(frame):
    x.append(x[-1] + 1)
    y.append(randrange(0, 10))
 
    ln.set_data(x, y) 
    return ln,
 
animation = FuncAnimation(fig, update, interval=500)
plt.show()
# %%
