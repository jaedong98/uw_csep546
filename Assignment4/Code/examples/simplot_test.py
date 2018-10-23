import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)
s2 = 2 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)
ax.plot(t, s2)
ax.legend(('t1', 't2'))

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks',
       )
ax.grid()


def draw_weights(iters, weights, xlabel, ylabel, title, img_fname):
    fig, ax = plt.subplots()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    for ws in zip(*weights):
        ax.plot(iters, ws)

    ax.legend(('w0', 'w1', 'w2', 'w3', 'w4'))
    ax.grid()
    plt.show()

draw_weights([1,2,3], [[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]], 'Iterations', 'Weights', 'Weights', 'png')