import numpy as np

from .simulation import Simulation


class DescendLoopException(Exception):
    pass


def gradient_descent(simulation, iter_max=50, step_init=1, tolerance=10**-9):
    '''Runs the gradient descent procedure.

    Parameters:
        simulation: Simulation
            Simulation object used as the initial guess.
        iter_max: integer
            The maximal allowed number of major iterations.
        step_init: float
            The initial value of the descent step (Dj multiplier).
        tolerance: float
            The gradient descent procedure stops when the gradient norm becomes
            less than tolerance.

    Returns:
        descent: [Simulation]
            List of simulations, corresponding to succesful steps of gradient
            descent, starting with the provided initial guess.

    '''

    try:
        print('Starting the gradient descent procedure...')
        descent = [simulation]
        step = step_init
        problem = simulation.problem

        print(f'{"i.j " :>6} {"s":>14} {"J":>14} {"norm(Dj)":>14}')
        print(f'{simulation.J:36.7e} {simulation.Dj_norm:14.7e}')

        for i in range(iter_max):
            if simulation.Dj_norm < tolerance:
                print(f'norm(Dj) = {simulation.Dj_norm:.7e} < tolerance')
                break

            j = 0
            while True:
                control_trial = (
                    simulation.control - step * simulation.Dj).clip(0, 1)
                if np.allclose(control_trial, simulation.control):
                    raise DescendLoopException
                simulation_trial = simulation.spawn(control_trial)

                if simulation_trial.J < simulation.J: break
                print(f'{i:3}.{j:<2} {step:14.7e} '
                      f'{simulation_trial.J:14.7e}  {13*"-"}')
                j += 1
                step /= 2

            simulation = simulation_trial
            print(f'{i:3}.{j:<2} {step:14.7e} '
                  f'{simulation.J:14.7e} {simulation.Dj_norm:14.7e}')
            descent.append(simulation)
            step *= 2

        else:
            print('Maximal number of iterations was reached.')

    except KeyboardInterrupt:
        print('Interrupted by user...')
    except DescendLoopException:
        print('The descend procedure has looped since control reached '
              'the boundary of the feasible set:\n'
              'project(control_next) == control_current.')

    print('Terminating.')

    return descent


def gradient_test(simulation,
                  iter_max=15, eps_init=10**-6, diff_type='forward'):
    '''Checks the accuracy of the calculated gradient Dj.

    The finite difference for J w.r.t. a random normalized direction is compared
    to the inner product (Dj,direction),  the absolute and the relative errors
    are calculated.

    Every iteration epsilon is divided by two.

    Parameters:
        simulation: Simulation
            Initial simulation used for testing.
        n: integer
            Number of tests.
        diff_type: 'forward' (default) or 'two_sided'
            The exact for of the finite difference expression.
        eps_init: float
            The initial value of epsilon.

    Returns:
        epsilons: [float]
            Epsilons used for testing.
        deltas: [float]
            Relative errors.

    '''

    print('Starting the gradient test...')

    Nt = simulation.problem.time_domain.Nt
    dt = simulation.problem.time_domain.dt
    # np.random.seed(0)
    direction = np.random.rand(Nt)
    direction /= simulation.problem.norm(direction)

    problem = simulation.problem
    D = simulation.Dj

    epsilons = []
    deltas = []

    inner_product = dt * np.sum(D*direction)
    print(f'    inner(Dj,direction) = {inner_product:.8e}')
    print(f'{"epsilon":>20}{"diff":>20}{"delta_abs":>20}{"delta":>20}')

    try:
        for eps in (eps_init * 2**-k for k in range(iter_max)):
            if diff_type == 'forward':
                control_ = (simulation.control + eps * direction).clip(0, 1)
                simulation_ = simulation.spawn(control_)
                diff = (simulation_.J - simulation.J) / eps

            elif diff_type == 'two_sided':
                control_ = (control - eps * direction).clip(0, 1)
                simulation_ = simulation.spawn(control_)
                control__ = (control + eps * direction).clip(0, 1)
                simulation__ = simulation.spawn(control__)
                diff = (simulation__.J - simulation_.J) / (2 * eps)

            delta_abs = inner_product - diff
            delta = delta_abs / inner_product
            deltas.append(delta)
            epsilons.append(eps)

            print(f'{eps:20.8e}{diff:20.8e}{delta_abs:20.8e}{delta:20.8e}')

        print('Maximal number of iterations was reached.')

    except KeyboardInterrupt:
        print('Interrupted by user...')

    print('Terminating.')

    return epsilons, deltas
