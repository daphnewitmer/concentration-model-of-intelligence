import matplotlib.pyplot as plt


def plot_distribution(x, y=None, title='', x_title='', y_title=''):
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()