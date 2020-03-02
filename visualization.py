import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
# from matplotlib import colors


def control_plot(reference, initial, optimal):

    fig, ax = plt.subplots()
    ax.set_title('Control optimization')

    x_dots = np.arange(len(reference))
    ax.set_xticks(x_dots)

    ax.scatter(x_dots, reference, color='purple', zorder=1, label='Reference control')
    ax.plot(reference, color='purple', zorder=1)
    ax.scatter(x_dots, initial, color='gray', zorder=2, alpha=0.5, label='Initial guess')
    ax.plot(initial, color='gray', zorder=2, alpha=0.5)
    ax.scatter(x_dots, optimal, color='blue', zorder=3, label='Optimized')
    ax.plot(optimal, color='blue', zorder=3)

    ax.fill_between(x_dots, optimal, reference, alpha=0.2, color='blue')

    ax.legend(loc=2)

    plt.tight_layout()
    plt.show()

def gradient_test_plot(values_eps, values_delta):

    fig, ax = plt.subplots()
    ax.set_title('Gradient test')

    # x_dots = np.arange(len(reference))
    # ax.set_xticks(x_dots)

    values_delta = [abs(x) for x in values_delta]

    ax.set_ylim([0,1.1 * max(values_delta)])

    # ax.scatter(values_eps, values_delta, color='red', zorder=2)
    ax.plot(values_eps, values_delta, color='blue', zorder=1)

    ax.legend(loc=2)

    plt.tight_layout()
    plt.show()

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