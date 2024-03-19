import argparse
import numpy as np
import os


def postprocess_bitstrings(dist: list[dict], grad: bool = False) -> np.ndarray:
    """
    Classical post-processing with maximal globality, as described in
    "Quantum Policy Gradient Algorithm with Optimized Action Decoding", Meyer et al., 2022

    :param dist: bitstring probabilities
    :param grad: whether this is for policy or gradient computation
    :return: expectation value used for policy construction
    """

    prob_odd_parity = 0
    for d in dist:
        if 1 == bin(d)[2:].count('1') % 2:
            prob_odd_parity += dist[d]
    if grad:
        return np.array([-prob_odd_parity, prob_odd_parity])
    else:
        return np.array([1-prob_odd_parity, prob_odd_parity])


def select_action(policy: np.ndarray) -> int:
    """
    Sample action according to given policy

    :param policy: probability distribution to sample from
    :return: sampled action
    """

    # clipping for some numerical errors where the probabilities would become slightly negative
    return np.random.choice(np.array([0, 1]), p=np.clip(policy, 0.0, 1.0))


def compute_loss(selected_policy: float, selected_gradients: np.ndarray, reward: float) -> np.ndarray:
    """
    Compute loss as $\Delta ln \pi(a | s) * reward = (\Delta \pi(a | s)) / (\pi(a | s)) * reward$

    :param selected_policy: policy value associated with executed action
    :param selected_gradients: gradients values associated with executed action
    :param reward: received reward
    :return: loss for all parameters
    """

    return (selected_gradients / selected_policy) * reward


def compute_natural_loss(selected_policy: float, selected_policy_gradients: np.ndarray, reward: float,
                         inverted_qfi: np.ndarray) -> np.ndarray:
    """
    Compute natural loss as $\Delta ln \pi(a | s) * reward * g^-1 = (\Delta \pi(a | s)) / (\pi(a | s)) * reward * g^-1$

    :param selected_policy: policy value associated with executed action
    :param selected_policy_gradients: gradients values associated with executed action
    :param reward: received reward
    :param inverted_qfi: inverted diagonal approximation of QFIM
    :return: natural loss for all parameters
    """

    log_policy_gradients = selected_policy_gradients / selected_policy
    # Apply inverted QFI
    natural_log_policy_gradients = log_policy_gradients * inverted_qfi

    return natural_log_policy_gradients * reward


def invert_qfi(qfi: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Invert diagonal matrix (just invert diagonal elements)

    :param qfi: diagonal elements of QFIM
    :param epsilon: small addition to prevent division by 0
    :return: inverted QFIM
    """

    shifted_qfi = qfi + epsilon
    inverted_qfi = [1/sq for sq in shifted_qfi]
    return np.array(inverted_qfi)


def generate_path(args: argparse.Namespace) -> (str, str):
    """
    Generate path to store results

    :param args: arguments
    :return: path, name
    """

    experiment_path = 'size={}qubits'.format(args.num_qubits)
    if args.path != '':
        experiment_path += '_{}'.format(args.path)
    experiment_name = 'g={}_lr={}_bs={}_init={}_'.format(args.gradient, args.learning_rate, args.batch_size, args.initializer)
    if args.experiment_suffix != '':
        experiment_name += '{}'.format(args.experiment_suffix)
    else:
        index = 0
        while True:
            test_path = 'tensorboard/{}/{}{}'.format(experiment_path, experiment_name, index)
            if not os.path.exists(test_path):
                experiment_name += '{}'.format(index)
                break
            index += 1
    return experiment_path, experiment_name


def parse_args() -> argparse.Namespace:
    """
    Parse input arguments

    :return: argument namespace
    """

    parser = argparse.ArgumentParser(description='Quantum Natural Policy Gradient (QNPG) Algorithm')

    _gradient_choices = ['first_order', 'natural']
    _qfi_approximation_choices = ['diag', 'block_diag']

    parser.add_argument('--gradient', '-grad', default='first_order', type=str,
                        choices=_gradient_choices,
                        help='Gradient methods to use:' +
                             ' | '.join(_gradient_choices) +
                             ' (default: first_order)')

    parser.add_argument('--spsa', '-spsa', action='store_true',
                        help='Use SPSA estimation of first-order gradients')

    parser.add_argument('--qfi_approximation', '-qfi_approx', default='diag', type=str,
                        choices=_qfi_approximation_choices,
                        help='Approximation to use for evaluating qfi-circuits:' +
                             ' | '.join(_qfi_approximation_choices) +
                             ' (default: diag)')

    parser.add_argument('--num_qubits', '-nq', default=12, type=int, metavar='N',
                        help='Number of qubits to use, has to be even for implemented circuits (default: 12)')

    parser.add_argument('--shots', '-shots', default=1024, type=int, metavar='N',
                        help='Number of shots to use. Set to `0` for exact computation (default: 1024)')

    parser.add_argument('--shots_grad', '-shots_grad', default=0, type=int, metavar='N',
                        help='Number of shots to use for (first and second order) gradient evaluation'
                             '(default: 0, i.e. `--shots` is used)')

    parser.add_argument('--learning_rate', '-lr', default=0.1, type=float, metavar='F',
                        help='Learning rate to use for variational parameters (default: 0.1)')

    parser.add_argument('--initializer', '-init', default=1.1, type=float, metavar='F',
                        help='Initializes to a N(mu=0.5, sigma=1)-neighborhood of optimal promising parameters.'
                             'For values > 1 values are sampled uniformly at random from [0, 2*pi] (default: 1.1)')

    parser.add_argument('--initializer_seed', '-init_seed', default=-1, type=int, metavar='N',
                        help='Random seed for parameter initialization, -1 for no seed (default: -1)')

    parser.add_argument('--batch_size', '-bs', default=10, type=int, metavar='N',
                        help='Batch size to use for training (default: 10)')

    parser.add_argument('--episodes', '-ep', default=1000, type=int, metavar='N',
                        help='Number of episodes to train for (default: 1000)')

    parser.add_argument('--validation_interval', '-val_int', default=1, type=int, metavar='N',
                        help='Perform validation at each `--validation_interval` batch updates (default: 1)')

    parser.add_argument('--validation_samples', '-val_sam', default=16, type=int, metavar='N',
                        help='Number of distinct samples (capped by  2**num_qubits) for validation (default: 16)')

    parser.add_argument('--path', '-path', default='', type=str,
                        help='Path for saving results relative to `./results.` (default: ``)')

    parser.add_argument('--experiment_suffix', '-suffix', default='', type=str,
                        help='Suffix to append to experiment logging (default: ``)')

    args = parser.parse_args()

    if not 0 == args.num_qubits % 2:
        raise ValueError('The implemented circuit construction assumes, that the number of qubits is even.')

    if args.initializer < 0:
        raise ValueError('The value of `initializer` has to be non-negative!')

    if 0 == args.shots:
        args.shots = None
        args.shots_grad = None

    if args.validation_samples > 2 ** args.num_qubits:
        print('[Argument Parser] Reduced `validation_samples` to `{}`, as these is the maximum number of distinct '
              'states for the selected system size.'.format(2 ** args.num_qubits))
        print()
        args.validation_samples = 2 ** args.num_qubits

    # do not select any random seed
    if -1 == args.initializer_seed:
        args.initializer_seed = None

    if 0 == args.shots_grad:
        args.shots_grad = args.shots

    return args
