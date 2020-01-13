import numpy as np
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
