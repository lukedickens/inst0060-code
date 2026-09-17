import numpy as np
import matplotlib.pyplot as plt

def plot_train_test_errors(
        control_var, experiment_sequence, train_errors, test_errors):
    """
    inputs
    ------
    control_var - the name of the control variable, e.g. degree (for polynomial)
        degree.
    experiment_sequence - a list of values applied to the control variable.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    train_line, = ax.plot(experiment_sequence, train_errors,'b-')
    test_line, = ax.plot(experiment_sequence, test_errors, 'r-')
    ax.set_xlabel(control_var)
    ax.set_ylabel("$E_{RMS}$")
    ax.legend([train_line, test_line], ["train", "test"])
    # errors won't strictly lie below 1 but very large errors can make the plot
    # difficult to read. So we restrict the limits of the y-axis.
    ax.set_ylim((0,1))
    return fig, ax

def plot_function(true_func, xlim=(0,1), ylim=(0,1), padding=0.05):
    fig, ax = plt.subplots()
    xs = np.linspace(xlim[0], xlim[1], 101)
    true_ys = true_func(xs)
    ax.plot(xs, true_ys, 'g-', linewidth=3)
    # set the x bounds with a bit of padding
    ax.set_xlim(xlim[0]-padding, xlim[1]+padding)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig, ax

def plot_function_and_data(inputs, targets, true_func):
    fig, ax = plot_function(true_func)
    ax.plot(inputs, targets, 'bo', markersize=3)
    return fig, ax

def plot_function_data_and_approximation(
        predict_func, inputs, targets, true_func):
    fig, ax = plot_function_and_data(inputs, targets, true_func)
    xs = np.linspace(0, 1, 101)
    ys = predict_func(xs)
    ax.plot(xs, ys, 'r-', linewidth=3)
    return fig, ax


