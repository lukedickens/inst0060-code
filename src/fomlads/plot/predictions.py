import numpy as np
import matplotlib.pyplot as plt

def plot_function(true_func, linewidth=3, xlim=None):
    """
    Plot a function in a given range

    parameters
    ----------
    true_func - the function to plot
    xlim (optional) - the range of values to plot for. A pair of values lower,
        upper. If not specified, the default will be (0,1)
    linewidth (optional) - the width of the plotted line

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the (one) line objects on the plot
    """
    if xlim is None:
        xlim = (0,1)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    xs = np.linspace(xlim[0], xlim[1], 101)
    true_ys = true_func(xs)
    line, = ax.plot(xs, true_ys, 'g-', linewidth=linewidth)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.5, 1.5)
    return fig, ax, [line]

def plot_function_and_data(inputs, targets, true_func, markersize=5, **kwargs):
    """
    Plot a function and some associated regression data in a given range

    parameters
    ----------
    inputs - the input data
    targets - the targets
    true_func - the function to plot
    markersize (optional) - the size of the markers in the plotted data
    <for other optional arguments see plot_function>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
    fig, ax, lines = plot_function(true_func)
    line, = ax.plot(inputs, targets, 'bo', markersize=markersize)
    lines.append(line)
    return fig, ax, lines

def plot_function_data_and_approximation(
        predict_func, inputs, targets, true_func, linewidth=3, xlim=None,
        **kwargs):
    """
    Plot a function, some associated regression data and an approximation
    in a given range

    parameters
    ----------
    predict_func - the approximating function
    inputs - the input data
    targets - the targets
    true_func - the true function
    <for optional arguments see plot_function_and_data>

    returns
    -------
    fig - the figure object for the plot
    ax - the axes object for the plot
    lines - a list of the line objects on the plot
    """
    if xlim is None:
        xlim = (0,1)
    fig, ax, lines = plot_function_and_data(
        inputs, targets, true_func, linewidth=linewidth, xlim=xlim, **kwargs)
    xs = np.linspace(0, 1, 101)
    ys = predict_func(xs)
    line, = ax.plot(xs, ys, 'r-', linewidth=linewidth)
    lines.append(line)
    return fig, ax, lines

