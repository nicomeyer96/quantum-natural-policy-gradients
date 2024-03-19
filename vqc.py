from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

import numpy as np


def generate_circuit(num_qubits: int) -> (QuantumCircuit, ParameterVector, ParameterVector):
    """
    Generates a scalable version of the circuit depicted in Fig. 5 (a) of
    Meyer et al., "Quantum Natural Policy Gradients: Towards Sample-Efficient Reinforcement Learning", arXiv:XXXX.XXXXX (2023).

    :param num_qubits: number of qubits to scale to (has to be even)
    :return: VQC, learnable parameters handle, encoding parameters handle
    """

    vqc = QuantumCircuit(num_qubits)

    params_encoding = ParameterVector('s', length=num_qubits)
    params_variational = ParameterVector('\u03B8', length=3 * num_qubits)

    # create initial equal superposition
    for q in range(num_qubits):
        vqc.h(q)

    # encode 0 as |R> = 1/sqrt(2) [1 i] and 1 as |L> = 1/sqrt(2) [1 -i]
    for q in range(num_qubits):
        vqc.p(np.pi/2 * (1 - 2 * params_encoding[q]), q)

    vqc.barrier()

    # first variational layer
    for q in range(num_qubits):
        if 0 == q % 2:
            vqc.rx(params_variational[q], q)
        else:
            vqc.rz(params_variational[q], q)

    # first entanglement layer with even qubits as control
    for q in range(num_qubits):
        if 0 == q % 2:
            vqc.cx(q, q + 1)

    vqc.barrier()

    # second variational layer
    for q in range(num_qubits):
        vqc.ry(params_variational[num_qubits + q], q)

    # second entanglement layer with odd qubits as control
    for q in range(num_qubits):
        if 1 == q % 2:
            vqc.cx(q, (q + 1) % num_qubits)

    vqc.barrier()

    # third variational layer
    for q in range(num_qubits):
        vqc.ry(params_variational[2 * num_qubits + q], q)

    return vqc, params_variational, params_encoding
