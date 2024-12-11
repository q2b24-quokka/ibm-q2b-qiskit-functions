#!/usr/bin/python3

import math
from qiskit import *
from qiskit.compiler import transpile
from qiskit.circuit.library import MCMT, ZGate
from qiskit_aer import AerSimulator
import numpy as np
from shss.utils import *

class SemanticHilbertSpaceSearchCustom:    
    def __init__(self, n):
        self.circuits = []
        self.ionq_circuits = []
        self.Q = math.ceil(np.log(n)/np.log(2)) # number of qubits
        self.N = 2**self.Q                      # number of discrete states (num sems rounded to power of 2)
        # self.ITER = 2
        self.ITER = math.ceil(math.pi/4*np.sqrt(self.N))-2 | 1

    def _amplitudes(self, doc):
        amplitude = np.sqrt(1/len(doc))
        return [amplitude if k in doc else 0 for k in range(self.N)]
    
    def set_query(self, corpus, query):
        qubits = [q for q in range(self.Q)]
        self.circuits = []
        V = self._phaseflip_gate(v=query, anti=False, name='V')
        W = self._W_gate()
        
        for doc in corpus:
            qc = QuantumCircuit(self.Q)
            qc.initialize(self._amplitudes(doc), qubits)
            qc.barrier()

            Iz = self._phaseflip_gate(v=doc, anti=False, name='Iz')
            Idqz = self._phaseflip_gate(v=doc.difference(query), anti=False, name='Id-qz')
            Iqz = self._phaseflip_gate(v=query, anti=True, name='I-qz')
            
            qc.append(V, qubits)
            qc.append(W, qubits)
            qc.append(Iz, qubits)
            qc.append(W, qubits)
            qc.append(Idqz, qubits)
            qc.barrier()

            for i in range(self.ITER):
                qc.append(V, qubits)
                qc.append(W, qubits)
                qc.append(Iqz, qubits)
                qc.barrier()

            qc.measure_all()
            self.circuits.append(qc)
    
    def _phaseflip_gate(self, v, anti=False, name=None):
        qc = QuantumCircuit(self.Q)
        matrix = []
        for i in range(self.N):
            row = [0] * (self.N)
            if i in v:
                phase = 1 if anti else -1
                row[i] = phase
            else:
                phase = -1 if anti else 1
                row[i] = phase
            matrix.append(row)
        qc.unitary(matrix, [q for q in range(self.Q)])
        ugate = qc.to_gate()
        if name:
            ugate.name = name
        return ugate

    # ATTRIBUTION: extends qiskit
    def _W_gate(self):
        qc = QuantumCircuit(self.Q)
        # stateprep 
        for qubit in range(self.Q):
            qc.h(qubit)
        for qubit in range(self.Q):
            qc.x(qubit)

        # multi-controlled-Z gate
        qc.h(self.Q - 1)  
        mcmt = MCMT(ZGate(), num_ctrl_qubits=self.Q - 1, num_target_qubits=1)  
        qc.compose(mcmt, list(range(self.Q)), inplace=True)  # Apply MCMT
        qc.h(self.Q - 1)  

        # stateprep+
        for qubit in range(self.Q):
            qc.x(qubit)
        for qubit in range(self.Q):
            qc.h(qubit)
        W = qc.to_gate()
        W.name = 'W'
        return W
    
    def _transpile_circuits(self, backend):
        circs = []
        for idx, qc in enumerate(self.circuits):
            print(f'DOCUMENT {idx+1}: transpiling circuit')
            t = transpile(qc, backend)
            circs.append(t)
        return circs
    
    def run(self, shots, backend='qasm_simulator'):
        print(f'Backend: {backend} | Num qubits: {self.Q} | Num iterations: {self.ITER}')
        exp_results = []
        match backend:
            case 'qasm_simulator':
                simulator = AerSimulator()
                for idx, qc in enumerate(self.circuits):
                    print(f'DOCUMENT {idx + 1}: transpiling circuit')
                    compiled_circuit = transpile(qc, simulator)  # Transpile for the simulator

                    print(f'DOCUMENT {idx + 1}: running simulation')
                    sim_result = simulator.run(compiled_circuit, shots=shots).result()  # Run the simulation

                    exp_results.append(sim_result)
                return exp_results
            # case 'ionq_simulator' | 'ionq_qpu':
            #     provider = IonQProvider()
            #     backend = provider.get_backend(backend)
            #     self.ionq_circuits = self._transpile_circuits(backend)
            #     for idx, qc in enumerate(self.ionq_circuits):
            #         job = backend.run(qc, shots=shots)
            #         print(f'DOCUMENT {idx+1}: running job {job.job_id()}')
            #         result = job.result()
            #         exp_results.append(result)
            #     return exp_results
            # case 'ibmq_qasm_simulator':
            #     provider = IBMQ.load_account()
            #     service = QiskitRuntimeService(channel='ibm_quantum')
            #     with Session(service=service, backend=backend):
            #         sampler = Sampler()
            #         job = sampler.run(circuits=self.circuits, shots=100)
            #         print(f'Running job {job.job_id()}')
            #         result = job.result()
            #         return result
            case _:
                raise Exception(f'No matching backend for {backend}')