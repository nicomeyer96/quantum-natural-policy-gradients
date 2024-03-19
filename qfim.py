import copy
import numpy as np

from qiskit.primitives import BaseSampler
from qiskit.providers import Job

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import SdgGate, HGate

from qiskit.dagcircuit import DAGOpNode, DAGOutNode, DAGInNode
from qiskit.converters import circuit_to_dag, dag_to_circuit


class SamplerQFI:
    """
    Class for computing/sampling the Quantum Fisher Information Matrix (QFIM).
    It uses a (block-)diagonal approximation, as described in
    Stokes et al. "Quantum Natural Gradient", Quantum 4, 269 (2020).
    """

    def __init__(self,
                 policy_circuit: QuantumCircuit,
                 parameters: ParameterVector,
                 sampler: BaseSampler,
                 shots: int | None = 1024,
                 mode: str = 'diag'):
        """
        Set up approximation of QFIM

        :param policy_circuit: circuit that described the policy
        :param parameters: free parameters w.r.t. which the QFIM should be computed
        :param sampler: sampler primitive to evaluate results
        :param shots: number of shots for estimating expectation values
        :param mode: diagonal or block-diagonal
        """

        self.policy_circuit = policy_circuit
        self.parameters = parameters
        self.sampler = sampler
        self.shots = shots
        self.mode = mode

        self.num_layers = None
        self.num_qubits = policy_circuit.num_qubits

        self.qfi_circuits = self._generate_qfi_circuits(policy_circuit)

    def _generate_qfi_circuits(self, policy_circuit: QuantumCircuit) -> list[QuantumCircuit]:
        """
        Construct the circuits for a (block-)diagonal approximation of the QFIM.

        NOTE: In principle, the results should be equivalent to qiskit.opflow.gradients.NaturalGradient.
              However, there seem to be some errors in this implementation, as the circuits deviate slightly for
              depths >= 2 (tested with qiskit version 0.41.1).

        :param policy_circuit: circuit
        :return: layer-wise list of qfi circuits
        """

        print('             _______________________________________________________________________________________________________')
        print('             |  Make sure the parameters are ordered from upper to lowermost wire and overall from left to right.  |')
        print('             |  Also it is required that each parameterized layer contains parameterized gates on all qubits:      |')
        print('             |                                                                                                     |')
        print('[SamplerQFI] |            --|0|----|3|----|6|--           --|0|----|1|----|2|--      --|0|----|3|----|5|--         |')
        print('             |  Correct:  --|1|----|4|----|7|--   Wrong:  --|3|----|4|----|5|--  or  --|1|----|4|----|6|--         |')
        print('             |            --|2|----|5|----|8|--           --|6|----|7|----|8|--      --|2|-----------|7|--         |')
        print('             |_____________________________________________________________________________________________________|')
        print()

        # convert to directed acyclic graph to ensure not violating dependencies
        policy_dag = circuit_to_dag(policy_circuit)

        # start with full policy graph and work from end to front
        qfi_dags = [policy_dag]

        # identify topology / layers of dag for easier iteration
        layers = list(policy_dag.multigraph_layers())[::-1]

        # iterate over layers from back to front
        for layer in layers:

            # if we did not encounter any parameterized gate in this layer we can just continue
            flag_found_parameterized_gate = False

            # iterate over individual gates in current layer
            for node in layer:

                # if Input or Output this is not relevant for us, so just continue
                if isinstance(node, DAGOutNode) or isinstance(node, DAGInNode):
                    continue

                # here it gets interesting, as these are actual gates acting on the qubits
                elif isinstance(node, DAGOpNode):
                    params_current_gate = node.op.params

                    # encountered non-parameterized gate
                    if 0 == len(params_current_gate):

                        # if this flag is already set to true, we encountered a parameterized gate in this layer
                        # -> added a new circuit/dag to qfi_dags
                        # -> do not need to consider this gate in second-to-last circuit in qfi_dags (could also leave it there, does not really matter)
                        if flag_found_parameterized_gate:
                            qfi_dags[-2].remove_op_node(node)

                        # remove also from newely appended layer (i.e. the one that will be investigated in the next iteration)
                        qfi_dags[-1].remove_op_node(node)

                    # encountered parameterized gate
                    elif 1 == len(params_current_gate):

                        # there might be only bound parameters, or we do not want to differentiate (e.g. encoding)
                        # -> we can remove it and continue
                        if 0 == len(params_current_gate[0].parameters) \
                                or list(params_current_gate[0].parameters)[-1] not in self.parameters:
                            qfi_dags[-1].remove_op_node(node)
                            continue

                        # this is the first parameterized gate in this layer, so we need to add a new element to qfi_dags
                        # and set the flag to true
                        if not flag_found_parameterized_gate:
                            qfi_dags.append(copy.deepcopy(qfi_dags[-1]))
                            flag_found_parameterized_gate = True

                        # we need to perform some change of basis, depending on the kind of parameterized rotation that is encountered
                        # (see Fig. 2 of "Quantum Natural Policy Gradients for Contextual Bandits", N. Meyer et al., 2023)
                        gate_type = node.name
                        if 'rz' == gate_type:
                            qfi_dags[-1].remove_op_node(node)
                        elif 'ry' == gate_type:
                            qfi_dags[-1].remove_op_node(node)
                            qfi_dags[-2].apply_operation_back(SdgGate(), node.qargs)
                            qfi_dags[-2].apply_operation_back(HGate(), node.qargs)
                        elif 'rx' == gate_type:
                            qfi_dags[-1].remove_op_node(node)
                            qfi_dags[-2].apply_operation_back(HGate(), node.qargs)
                        else:
                            raise NotImplementedError('Only parameterized `Rx`, `Ry`, and `Rz` rotations are currently supported!')
                    else:
                        raise NotImplementedError('Multi-parameterized gates are currently not supported!')
                else:
                    raise NotImplementedError()

        # convert back to circuits
        qfi_circuits = [dag_to_circuit(d) for d in qfi_dags]

        # last element might potentially contain no free parameters due to construction above -> remove
        if 0 == len(qfi_circuits[-1].parameters):
            qfi_circuits.pop()

        print("[BaseQFI] Identified {} layers.".format(len(qfi_circuits)))
        print()

        self.num_layers = len(qfi_circuits)

        # invert list to ensure correct association with ordering of parameters (i.e. left to right)
        qfi_circuits = qfi_circuits[::-1]

        # Append measurements to all qubits.
        # CAUTION! Here we assume, that each qubit contains a parameterized gate in each layer,
        # which usually is the case for the typical VQC architectures
        for qfi_circuit in qfi_circuits:
            qfi_circuit.measure_all(inplace=True)

        print("[SamplerQFI] Composed {} circuits for evaluating the QFI.".format(len(qfi_circuits)))
        print()

        return qfi_circuits

    def print_mode(self):
        """
        Print mode of QFIM approximation (diagonal or block-diagonal)
        """

        print(self.mode)

    def print_qfi_circuits(self):
        """
        Print circuits for approximating QFIM
        """

        for qfi_circuit in self.qfi_circuits:
            print(qfi_circuit)

    def get_number_layers(self):
        """
        Get number of layers (equals number of blocks of QFIM)
        """

        return self.num_layers

    def _run_single(self,
                    param_values: np.ndarray,
                    params: ParameterVector,
                    params_fixed: ParameterVector) -> list[np.ndarray]:
        """
        Collect parameters for one batch element

        :param param_values: values of trainable parameters
        :param params: handles for trainable parameters
        :param params_fixed: handle for fixed parameters (e.g. encoding, where no derivative should be computed)
        :return: Learnable parameter values for one batch
        """

        # test whether number of free, fixed, and overall parameters are consistent
        number_given_params = len(params) + len(params_fixed)
        if not number_given_params == self.policy_circuit.num_parameters:
            raise RuntimeError('The number of given parameters ({} + {} = {}) does not match the number of parameters '
                               'in the circuit ({})'.format(len(params),
                                                            len(params_fixed),
                                                            len(params) + len(params_fixed),
                                                            self.policy_circuit.num_parameters))

        # cut only yhe necessary parameter values (as the QFIM requires execution of sub-circuits)
        params_qfi = []
        for i, layer in enumerate(range(self.num_layers)):
            # count the number of free parameters w.r.t. which the derivative should be taken
            num_parameters_layer = sum(el in self.qfi_circuits[i].parameters for el in params)
            params_layer = param_values[:num_parameters_layer]
            params_qfi.append(params_layer)

        return params_qfi

    def run(self,
            batch_param_values: np.ndarray | list[np.ndarray],
            batch_params: ParameterVector | list[ParameterVector],
            batch_param_values_fixed: np.ndarray | list[np.ndarray],
            batch_params_fixed: ParameterVector | list[ParameterVector]
            ) -> Job:
        """
        Set up job for QFIM estimation

        :param batch_param_values: values of variational parameters
        :param batch_params: handle for variational parameters
        :param batch_param_values_fixed: values of fixed parameters (e.g. encoding)
        :param batch_params_fixed: handles for fixed parameters (e.g. encoding)
        :return: job handle
        """

        # test whether batch sizes are consistent
        if not len(batch_param_values) == len(batch_params) == len(batch_param_values_fixed) == len(batch_params_fixed):
            raise RuntimeError('Error encountered while setting up batch-processing!')
        batch_size = len(batch_params)

        # iterate over batches to compose parameters (allows for sending everything to hardware/simulator at once)
        batch_params_qfi = []
        for param_values, params, params_fixed in zip(batch_param_values,
                                                      batch_params,
                                                      batch_params_fixed):
            batch_params_qfi.extend(self._run_single(param_values, params, params_fixed))

        # repeat circuits and fixed parameters for all batches
        batch_vqcs = []
        batch_param_values_fixed_repeated = []
        for b in range(batch_size):
            batch_vqcs.extend(self.qfi_circuits)
            for _ in range(len(self.qfi_circuits)):
                batch_param_values_fixed_repeated.append(batch_param_values_fixed[b])

        # runs all `batch_size x num_layers` circuits in one job
        job = self.sampler.run([v.bind_parameters({params_fixed: p}) for v, p
                                in zip(batch_vqcs, batch_param_values_fixed_repeated)],
                               batch_params_qfi,
                               shots=self.shots)

        return job

    def evaluate(self, job: Job, batch_size: int) -> list[np.ndarray] | list[list[np.ndarray]]:
        """
        Evaluate and post-process (block-)diagonal matrix

        :param job: previously constructed job object
        :param batch_size: number of elements in one batch
        :return: (block-)diagonal approximation of QFIM in sparse representation
        """

        # extract bistring counts
        batch_quasi_dists = job.result().quasi_dists

        batch_diags = []
        batch_exps = []

        length_of_batch = len(batch_quasi_dists) // batch_size
        if not len(batch_quasi_dists) == length_of_batch * batch_size:
            raise RuntimeError('Mismatch between provided batch_size and job object!')

        # iterate over batches
        for i in range(batch_size):

            # extract relevant information for current batch
            quasi_dists_current_batch = batch_quasi_dists[i*length_of_batch:(i+1)*length_of_batch]

            diags, exps = self._compute_diag(quasi_dists_current_batch)
            batch_diags.append(diags)
            batch_exps.append(exps)

        # return only diagonal elements in a 1D array of size `batch_size x num_parameters = batch_size x num_layers x num_qubits`
        # (where num_qubits corresponds to the number of parameterized gates per layer by requirement)
        if 'diag' == self.mode:
            batch_diag_elements = []
            for diags in batch_diags:
                diag_elements = np.reshape(diags, (len(diags)*len(diags[0]), ))
                batch_diag_elements.append(diag_elements)
            return batch_diag_elements

        # Compute also the non-diagonal block-diagonal elements
        batch_blocks = []
        for i in range(batch_size):
            quasi_dists_current_batch = batch_quasi_dists[i * length_of_batch:(i + 1) * length_of_batch]
            diags = batch_diags[i]
            exps = batch_exps[i]

            blocks = self._compute_block(quasi_dists_current_batch, diags, exps)
            batch_blocks.append(blocks)

        return batch_blocks

    def _compute_diag(self, quasi_dists: list[dict]) -> (np.ndarray, np.ndarray):
        """
        Compute diagonal elements of QFIM

        :param quasi_dists: bitstring counts
        :return: diagonal elements, associated expectation values
        """

        diags = []
        exps = []
        for quasi_dist in quasi_dists:
            diag, exp = self._compute_diag_layer(quasi_dist)
            diags.append(diag)
            exps.append(exp)
        return np.array(diags), np.array(exps)

    def _compute_diag_layer(self, quasi_dist: dict) -> (list[float], list[float]):
        """
        Extract the values for on block of the diagonal approximation (as variance of individual qubit measurements)

        :param quasi_dist: bitstring counts of this layer
        :return: diagonal elements, associated expectation values
        """

        diag_layer = []

        # of potential use for mode 'block_diag', so store it directly here
        exp_layer = []

        # convert to binary representation
        quasi_dist_bin = {}
        for q in quasi_dist:
            quasi_dist_bin[bin(q)[2:].zfill(self.num_qubits)] = quasi_dist[q]

        # iterate over individual bits to extract variances
        for n in range(self.num_qubits):
            prob_zero = 0
            for q in quasi_dist_bin:
                if '0' == q[n]:
                    prob_zero += quasi_dist_bin[q]
            exp_mean = 2 * prob_zero - 1
            exp_layer.append(exp_mean)
            exp_var = prob_zero * (1 - exp_mean) ** 2 + (1 - prob_zero) * (-1 - exp_mean) ** 2
            diag_layer.append(exp_var)

        return diag_layer, exp_layer

    def _compute_block(self, quasi_dists: list[dict], diags: list[list[float]], exps: list[list[float]]) -> np.ndarray:
        """
        Compute block-diagonal elements of QFIM

        :param quasi_dists: bitstring counts
        :param diags: diagonal elements previously computed
        :param exps: associated expectation values previously computed
        :return: block-diagonal approximation of QFIM in sparse representation
        """

        blocks = []
        for i, quasi_dist in enumerate(quasi_dists):
            block = self._compute_block_layer(quasi_dist, diags[i], exps[i])
            blocks.append(block)
        return np.array(blocks)

    def _compute_block_layer(self, quasi_dist: dict, diag: list[float], exp: list[float]):
        """
        Extract the values for on block of the block-diagonal approximation

        :param quasi_dist: bitstring counts
        :param diag: diagonal elements previously computed
        :param exp: associated expectation values previously computed
        :return: one block of block-diagonal approximation of QFIM
        """

        block = np.zeros((self.num_qubits, self.num_qubits))

        # insert diagonal elements
        for i in range(self.num_qubits):
            block[i, i] = diag[i]

        # binary representation of counts
        quasi_dist_bin = {}
        for q in quasi_dist:
            quasi_dist_bin[bin(q)[2:].zfill(self.num_qubits)] = quasi_dist[q]

        # iterate over upper block (excluding diagonals)
        for i in range(self.num_qubits):
            for j in range(i+1, self.num_qubits):
                # This entails measuring either `00` or `11`
                prob_zero_combined = 0
                for q in quasi_dist_bin:
                    if ('0' == q[i] and '0' == q[j]) or ('1' == q[i] and '1' == q[j]):
                        prob_zero_combined += quasi_dist_bin[q]
                exp_combined = 2 * prob_zero_combined - 1
                elem_combined = exp_combined - exp[i] * exp[j]
                block[i, j] = elem_combined
                # the block matrix is symmetric
                block[j, i] = elem_combined

        return block
