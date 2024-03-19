import os
import pickle
import numpy as np

from qiskit.primitives import Sampler
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.algorithms.gradients import ParamShiftSamplerGradient, SPSASamplerGradient

from torch.utils.tensorboard import SummaryWriter

from qfim import SamplerQFI
from vqc import generate_circuit
from training_helpers import Parameters, ProgressTracker
from environment import ContextualBandits, state_prep
from utils import parse_args, select_action, compute_loss, compute_natural_loss, invert_qfi, generate_path, postprocess_bitstrings


def train(args):
    """
    Train an QPG or QNPG agent, following concepts from:
    Jerbi et al., "Parametrized Quantum Policies for Reinforcement Learning", NeurIPS 34 (2021).
    Meyer et al., "Quantum Policy Gradient Algorithm with Optimized Action Decoding", arXiv:2212.06663 (2022).
    Meyer et al., "Quantum Natural Policy Gradients: Towards Sample-Efficient Reinforcement Learning", arXiv:XXXX.XXXXX (2023).

    :param args: arguments
    """

    # initialize environment depending on number of qubits
    env = ContextualBandits(n_arms=2, n_bandits=2**args.num_qubits)

    # construct VQC
    vqc, params_variational, params_encoding = generate_circuit(args.num_qubits)

    # initialize class for handling parameter initialization and updates
    params = Parameters(len(params_variational),
                        seed=args.initializer_seed,
                        lr=args.learning_rate,
                        initializer=args.initializer)

    # Qiskit Sampler primitive
    sampler_expectation = Sampler()
    if args.spsa:
        # in case SPSA-approximation of first-order gradients should be used
        sampler_gradient = SPSASamplerGradient(Sampler(), epsilon=0.1, batch_size=10, seed=0)
    else:
        # for first-order parameter-shift gradients
        sampler_gradient = ParamShiftSamplerGradient(Sampler())

    # initialize sampler for (block-)diagonal approximation of QFI (only if `--gradient=natural` is selected)
    sampler_qfi = None
    if 'natural' == args.gradient:
        sampler_qfi = SamplerQFI(vqc, params_variational, sampler_expectation,
                                 shots=args.shots_grad, mode=args.qfi_approximation)

    # Sampler primitive requires appending measurement to all (relevant) qubits
    # Important: Do after initializing SamplerQFI, as it cannot handle circuits with measurements
    vqc.measure_all(inplace=True)

    # determine number of batches to execute
    episodes_batch = args.episodes // args.batch_size

    progress = ProgressTracker()

    # generate paths, loggers, ..., for saving training progress and results
    experiment_path, experiment_name = generate_path(args)
    path = os.path.join(experiment_path, experiment_name)
    if not os.path.exists('results/{}'.format(path)):
        os.makedirs('results/{}'.format(path))
    logger = SummaryWriter('tensorboard/{}'.format(path))

    # store arguments
    with open('results/{}/{}'.format(path, 'arguments.npy'), 'wb') as ff:
        np.save(ff, args)

    # store initial parameter set
    with open('results/{}/{}'.format(path, 'params_0.npy'), 'wb') as ff:
        np.save(ff, params.get_params())

    print('[Trainer] Generated experiment path `{}`.'.format(path))
    print('[Trainer] Start training for {} batches with a batch size of {}.'.format(episodes_batch, args.batch_size))
    print()

    for episode_batch in range(episodes_batch):

        # This allows for evaluating all elements from a batch at once
        # (might be not the cleanest coding styles, but it tailored for the scheduling system on the IBMQ devices)
        batch_raw_state = [env.reset() for _ in range(args.batch_size)]

        # convert decimal state to (zero-padded) binary representation
        batch_state = [state_prep(rs, args.num_qubits) for rs in batch_raw_state]

        # submit jobs to sample policy for one batch (i.e. batch_size circuits)
        batch_job_expectation = sampler_expectation.run(
            [vqc.bind_parameters({params_encoding: s}) for s in batch_state],
            parameter_values=[params.get_params() for _ in range(args.batch_size)],
            shots=args.shots
        )

        # submit jobs to sample gradient of policy for one batch
        # (i.e. 2 * number_parameters * batch_size circuits for parameter-shift,
        # 2 * batch_size_spsa * batch_size for SPSA-approximation)
        batch_job_gradient = sampler_gradient.run(
            [vqc.bind_parameters({params_encoding: s}) for s in batch_state],
            parameter_values=[params.get_params() for _ in range(args.batch_size)],
            shots=args.shots_grad
        )

        # submit job to sample quantum QFI for one batch (i.e. number_layers * batch_size circuits)
        batch_job_qfi = None
        if 'natural' == args.gradient:
            batch_job_qfi = sampler_qfi.run(
                [params.get_params() for _ in range(args.batch_size)],
                [params_variational for _ in range(args.batch_size)],
                batch_state,
                [params_encoding for _ in range(args.batch_size)]
            )

        # extract the results
        batch_expectation_raw = batch_job_expectation.result().quasi_dists
        batch_gradient_raw = batch_job_gradient.result().gradients

        # post-process bitstring counts via global post-processing function
        batch_policy = np.array([postprocess_bitstrings(bs) for bs in batch_expectation_raw])
        batch_gradient = np.array(
            [[postprocess_bitstrings(bsg_partial, grad=True) for bsg_partial in bsg]
             for bsg in batch_gradient_raw]
        )

        # extract results for quantum QFI, invert diagonal matrix
        batch_inverted_qfi = None
        if 'natural' == args.gradient:
            batch_qfi = sampler_qfi.evaluate(batch_job_qfi, batch_size=args.batch_size)
            if 'diag' == args.qfi_approximation:
                batch_inverted_qfi = [invert_qfi(qfi) for qfi in batch_qfi]
            else:
                raise RuntimeError('Inverting the QFI is currently only supported for `qfi_approximation=diag`!')

        # select actions
        batch_action = [select_action(policy) for policy in batch_policy]

        batch_reward = []
        batch_expected_reward = []

        # Reset environment to corresponding state and execute action (this enables more efficient execution on IBMQ devices, as things can be batched)
        # (Giving the complete policy allows for computing exact expectation values of the reward, which could be used to track training progress.
        # This information is obviously NOT provided to the agent)
        for raw_state, action, policy in zip(batch_raw_state, batch_action, batch_policy):
            env.set_state(raw_state)
            _, reward, _, expected_reward = env.step(action=action, policy=policy)
            batch_reward.append(reward)
            batch_expected_reward.append(expected_reward)

        # some logging
        progress.add_batch_reward(batch_reward)
        progress.add_batch_expected_reward(batch_expected_reward)
        avg_batch_reward = np.average(np.array(batch_reward))
        avg_batch_expected_reward = np.average(np.array(batch_expected_reward))
        progress.add_reward(avg_batch_reward)
        progress.add_expected_reward(avg_batch_expected_reward)

        print('[Train <Episode {}> ({} samples)] Average reward = {:.3f} (Average expected reward = {:.3f})'
              .format(episode_batch * args.batch_size, args.batch_size,
                      avg_batch_reward,
                      avg_batch_expected_reward)
              )

        # some more logging via tensorboard
        for i in range(args.batch_size):
            logger.add_scalar('Episode/Reward', batch_reward[i], episode_batch * args.batch_size + i)
            logger.add_scalar('Episode/ExpectedReward', batch_expected_reward[i], episode_batch * args.batch_size + i)
        logger.add_scalar('Batch/Reward', np.average(batch_reward), episode_batch * args.batch_size)
        logger.add_scalar('Batch/ExpectedReward', np.average(batch_expected_reward), episode_batch * args.batch_size)

        # determine loss values
        losses = []
        if 'first_order' == args.gradient:
            # iterate over batch
            for policy, gradient, reward, action in zip(batch_policy,
                                                        batch_gradient,
                                                        batch_reward,
                                                        batch_action):
                losses.append(compute_loss(policy[action], gradient[:, action], reward))
        elif 'natural' == args.gradient:
            # iterate over batch
            for policy, gradient, reward, action, inverted_qfi in zip(batch_policy,
                                                                      batch_gradient,
                                                                      batch_reward,
                                                                      batch_action,
                                                                      batch_inverted_qfi):
                losses.append(compute_natural_loss(policy[action], gradient[:, action], reward, inverted_qfi))
        else:
            raise NotImplementedError('Gradient method {} is not implemented!'.format(args.gradient))

        # average over batch
        avg_losses = np.average(np.array(losses), axis=0)

        # update parameters with gradient ascent
        params.update_params(avg_losses)

        # perform validation
        if 0 == episode_batch % args.validation_interval:

            avg_reward_val, avg_expected_reward_val = validate(env, vqc, params_encoding, params, sampler_expectation,
                                                               args.num_qubits, args.shots, args.validation_samples)

            # some logging
            progress.add_validation_reward(avg_reward_val)
            progress.add_validation_expected_reward(avg_expected_reward_val)

            print('[Validate ({:.1f}% of state space)] Average reward = {:.3f} (Average expected reward = {:.3f})'
                  .format(100*args.validation_samples/(2**args.num_qubits),
                          avg_reward_val,
                          avg_expected_reward_val)
                  )

            # some more logging via tensorboard
            logger.add_scalar('Validation/Reward', np.average(batch_reward), (episode_batch + 1) * args.batch_size)
            logger.add_scalar('Validation/ExpectedReward', np.average(batch_expected_reward), (episode_batch + 1) * args.batch_size)
            with open('results/{}/{}'.format(path, 'params_{}.npy'.format((episode_batch+1)*args.batch_size)), 'wb') as ff:
                np.save(ff, params.get_params())

    # store the final parameters
    with open('results/{}/{}'.format(path, 'params_final.npy'), 'wb') as ff:
        np.save(ff, params.get_params())

    # store the overall training trajectory.
    history = progress.convert_to_dict()
    with open('results/{}/{}'.format(experiment_path, 'history_{}.pkl'.format(experiment_name)), 'wb') as ff:
        pickle.dump(history, ff)


def validate(env: ContextualBandits,
             vqc: QuantumCircuit,
             params_encoding: ParameterVector,
             params: Parameters,
             sampler: Sampler,
             num_qubits: int,
             shots_exp: int,
             number_samples: int) -> (float, float):
    """
    Validate current policy

    :param env: environment handle
    :param vqc: policy circuit
    :param params_encoding: encoding parameters handle
    :param params: variational parameters
    :param sampler: sampler primitive to estimate expectation values
    :param num_qubits: number of qubits in circuit
    :param shots_exp: shots to estimate expectation values with
    :param number_samples: number of random (disjunct) states to consider
    :return: average reward, average expected reward
    """

    # generate random states
    selected_states = np.random.permutation(2 ** num_qubits)[:number_samples]

    # convert to binary representation
    selected_states_bin = [state_prep(raw_state, num_qubits) for raw_state in selected_states]

    # set up job
    job_expectations = sampler.run(
        [vqc.bind_parameters({params_encoding: state}) for state in selected_states_bin],
        parameter_values=[params.get_params() for _ in range(number_samples)],
        shots=shots_exp
    )
    samples = job_expectations.result().quasi_dists
    policies = np.array([postprocess_bitstrings(s) for s in samples])
    actions = [select_action(policy) for policy in policies]

    # evaluate for all policies
    rewards = []
    expected_rewards = []
    for action, state, policy in zip(actions, selected_states, policies):
        env.set_state(state)
        _, reward, _, expected_reward = env.step(action, policy=policy)
        rewards.append(reward)
        expected_rewards.append(expected_reward)

    # average
    avg_reward = np.average(np.array(rewards))
    avg_expected_reward = np.average(np.array(expected_rewards))

    return avg_reward, avg_expected_reward


if __name__ == '__main__':
    train(parse_args())
