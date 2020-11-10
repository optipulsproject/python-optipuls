import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
# from matplotlib import colors


def control_plot(*controls, labels=None, outfile=None, **kwargs):

    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)

    ax.set_title('Control')

    x = np.arange(len(controls[0]))
    # ax.set_xticks(x)

    if not labels:
        labels = ['control ' + str(i+1) for i in range(len(controls))]

    for control, label in zip(controls, labels):
        # ax.scatter(x, objective, label=label, zorder=2)
        ax.scatter(x, control, color='purple', zorder=0, label=label)
        ax.plot(x, control, color='purple', zorder=1)

    ax.fill_between(x, controls[0], controls[-1], alpha=0.2, color='blue')

    ax.legend(loc=2)

    plt.tight_layout()
    if not outfile:
        plt.show(**kwargs)
    else:
        plt.savefig(outfile, **kwargs)



def objective_plot(*objectives, labels=None, outfile=None, **kwargs):

    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)

    ax.set_title('Objective in time')

    x = np.arange(len(objectives[0]))

    if not labels:
        labels = ['objective ' + str(i+1) for i in range(len(objectives))]

    for objective, label in zip(objectives, labels):
        # ax.scatter(x, objective, label=label, zorder=2)
        ax.bar(x, objective, label=label, alpha=0.3)

    ax.legend(loc=2)

    plt.tight_layout()
    if not outfile:
        plt.show(**kwargs)
    else:
        plt.savefig(outfile, **kwargs)


def gradient_test_plot(eps, *deltas, labels=None, outfile=None, **kwargs):
    '''Makes a plot of gradient_test output.

    Parameters:
        eps: array-like
            Epsilons used for testing.
        deltas: array-like, one or more
            Delta values to plot. Multiple given arrays provide multiple curves.
        labels: list of strings
            Labels for the given deltas.
        kwargs:
            Additional parameters for pyplot.show or pyplot.savefig.

    Usage:
        gradient_test_plot(eps, delta)
        gradient_test_plot(eps, delta, outfile='filename.png', dpi=160)
        gradient_test_plot(eps, delta_1, delta_2)
        gradient_test_plot(eps, delta_1, delta_2, labels=['label 1', 'label 2'])

    '''

    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_title('Gradient test')

    if not labels:
        labels = ['delta ' + str(i+1) for i in range(len(deltas))]

    for delta, label in zip(deltas, labels):
        ax.scatter(eps, np.abs(delta), label=label, zorder=2)
        ax.plot(eps, np.abs(delta), zorder=1)

    ax.legend(loc=2)

    plt.tight_layout()
    if not outfile:
        plt.show(**kwargs)
    else:
        plt.savefig(outfile, **kwargs)


def size_plot(time_space, r_sol, r_liq, d_sol, d_liq, sigma=3, scale=2000):
    dt = time_space[1] - time_space[0]
    fig, axes = plt.subplots(2,2)

    g_r_sol = scipy.ndimage.filters.gaussian_filter1d(r_sol, sigma)
    G_r_sol = np.gradient(g_r_sol, dt)
    axes[0,0].plot(time_space, r_sol[:-1]*scale, alpha=0.4, color='tab:blue')
    axes[0,0].plot(time_space, g_r_sol[:-1]*scale, color='tab:blue')
    axes[0,0].plot(time_space, G_r_sol[:-1], color='orange')
    axes[0,0].set_title('radius solidus')

    g_r_liq = scipy.ndimage.filters.gaussian_filter1d(r_liq, sigma)
    G_r_liq = np.gradient(g_r_liq, dt)
    axes[0,1].plot(time_space, r_liq[:-1]*scale, alpha=0.4, color='tab:blue')
    axes[0,1].plot(time_space, g_r_liq[:-1]*scale, color='tab:blue')
    axes[0,1].plot(time_space, G_r_liq[:-1], color='orange')
    axes[0,1].set_title('radius liquidus')

    g_d_sol = scipy.ndimage.filters.gaussian_filter1d(d_sol, sigma)
    G_d_sol = np.gradient(g_d_sol, dt)
    axes[1,0].plot(time_space, d_sol[:-1]*scale, alpha=0.4, color='tab:blue')
    axes[1,0].plot(time_space, g_d_sol[:-1]*scale, color='tab:blue')
    axes[1,0].plot(time_space, G_d_sol[:-1], color='orange')
    axes[1,0].set_title('depth solidus')

    g_d_liq = scipy.ndimage.filters.gaussian_filter1d(d_liq, sigma)
    G_d_liq = np.gradient(g_d_liq, dt)
    axes[1,1].plot(time_space, d_liq[:-1]*scale, alpha=0.4, color='tab:blue')
    axes[1,1].plot(time_space, g_d_liq[:-1]*scale, color='tab:blue')
    axes[1,1].plot(time_space, G_d_liq[:-1], color='orange')
    axes[1,1].set_title('depth liquidus')

    fig.subplots_adjust(hspace=0)
    plt.show()


def temperature_plot(timeline, *temperatures, labels=None, outfile=None,
                     hbars=None, **kwargs):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(12,8)

    ax.set_title('temperatures')
    ax.set_xticks(timeline)
    ax.set_ylim(0, 2000)

    # set dummy labels if they are not present
    if not labels:
        labels = ['temperature ' + str(i+1) for i in range(len(temperatures))]

    for temp, label in zip(temperatures, labels):
        ax.plot(timeline, temp, zorder=0, label=label)
        ax.scatter(timeline, temp, color='red', marker='x', zorder=1)

    if hbars is not None:
        for hbar in hbars:
            ax.axhline(hbar)

    ax.legend(loc=2)

    plt.tight_layout()
    if not outfile:
        plt.show(**kwargs)
    else:
        plt.savefig(outfile, **kwargs)




