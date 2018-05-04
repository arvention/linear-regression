import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Plotter(object):

    def __init__(self, x, y, ims):
        self.x = x
        self.y = y
        self.ims = ims

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(np.amin(x).item() - 1, np.amax(x).item() + 1)
        self.ax.set_ylim(np.amin(y).item() - 1, np.amax(y).item() + 1)
        plt.plot(self.x, self.y, 'ro', label='Original data')
        self.ln, = plt.plot([], [], label='Fitted line', animated=True)

    def animate(self, frame):
        plt.title('Epoch ' + str(frame[0]))
        x_data, y_data = [], []
        x_data.append(frame[1])
        y_data.append(frame[2])
        self.ln.set_data(x_data, y_data)
        return self.ln,

    def plot(self):
        plt.legend()
        anim = FuncAnimation(
            self.fig,
            self.animate,
            frames=self.ims,
            blit=True,
            repeat=True
        )

        anim.save('./animation.gif', writer='imagemagick', fps=5)
