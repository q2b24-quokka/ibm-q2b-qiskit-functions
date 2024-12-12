#!/usr/bin/python3

import math
from qiskit import *
from qiskit.compiler import transpile
from qiskit.circuit.library import MCMT, ZGate
from qiskit_aer import AerSimulator
import numpy as np
from shss.utils import *

class SemanticHilbertSpaceSearchNative:    
    def __init__(self, n):
        self.circuits = []
        self.ionq_circuits = []
        self.Q = math.ceil(np.log(n)/np.log(2)) # number of qubits
        self.N = 2**self.Q                      # number of discrete states (num sems rounded to power of 2)

    def _amplitudes(self, doc):
        amplitude = np.sqrt(1/len(doc))
        return [amplitude if k in doc else 0 for k in range(self.N)]
    
    def _query_idxs(self, query):
        if not isinstance(query, (set, list)):
            raise TypeError("Query must be a set or list of indices.")

        # convert `query` to a binary representation based on qubit indices
        qbin_format = f'0{self.Q}b'
        qstr = ''.join('1' if i in query else '0' for i in range(self.Q))
        qstr = qstr[::-1] 
        idxs = [idx for idx, b in enumerate(qstr) if b == '1']
        return idxs


    def set_query(self, corpus, query):
        Q_INPUT = self.Q
        Q_ANC = 1
        query_idxs = self._query_idxs(query)  # didnt use

        for doc in corpus:
            qr_doc = QuantumRegister(Q_INPUT, 'doc')
            qr_anc = QuantumRegister(Q_ANC, 'anc')
            cr_out = ClassicalRegister(Q_INPUT, 'out')
            qc = QuantumCircuit(qr_doc, qr_anc, cr_out)

            qc.initialize(self._amplitudes(doc), qr_doc)
            qc.initialize('1', qr_anc)
            qc.barrier()

            # V 
            qc.h(qr_anc[0])
            mcmt_v = MCMT(ZGate(), num_ctrl_qubits=Q_INPUT, num_target_qubits=1)
            qc.compose(mcmt_v, qr_doc[:] + [qr_anc[0]], inplace=True)
            qc.h(qr_anc[0]) 
            qc.barrier()

            # W 
            for qubit in qr_doc:
                qc.h(qubit)
            for qubit in qr_doc:
                qc.x(qubit)

            # Multi-controlled-Z using MCMT
            qc.h(qr_doc[-1]) 
            mcmt_w = MCMT(ZGate(), num_ctrl_qubits=Q_INPUT - 1, num_target_qubits=1)
            qc.compose(mcmt_w, qr_doc[:-1] + [qr_doc[-1]], inplace=True)
            qc.h(qr_doc[-1])  

            # cleam
            for qubit in qr_doc:
                qc.x(qubit)
            for qubit in qr_doc:
                qc.h(qubit)

            qc.measure(qr_doc, cr_out)
            self.circuits.append(qc)
    
    def _transpile_circuits(self, backend):
        circs = []
        for idx, qc in enumerate(self.circuits):
            print(f'DOCUMENT {idx+1}: transpiling circuit')
            t = transpile(qc, backend)
            circs.append(t)
        return circs
    
    def run(self, shots, backend='qasm_simulator'):
        print(f'Backend: {backend} | Num qubits: {self.Q}')
        exp_results = []
        match backend:
            case 'qasm_simulator':
                simulator = AerSimulator()
                for idx, qc in enumerate(self.circuits):
                    print(f'DOCUMENT {idx + 1}: transpiling circuit')
                    compiled_circuit = transpile(qc, simulator)

                    print(f'DOCUMENT {idx + 1}: running simulation')
                    sim_result = simulator.run(compiled_circuit, shots=shots).result()

                    exp_results.append(sim_result)
                return exp_results
            # case 'ibmq_qasm_simulator' | 'ibmq_belem' : # or least_busy
            #     provider = IBMQ.load_account()
            #     service = QiskitRuntimeService(channel='ibm_quantum')
            #     ibm_circuits = self._transpile_circuits(backend)
            #     with Session(service=service, backend=backend):
            #         sampler = Sampler()
            #         job = sampler.run(circuits=ibm_circuits, shots=shots)
            #         print(f'Running job {job.job_id()}')
            #         result = job.result()
            #         return result
            case _:
                raise Exception(f'No matching backend for {backend}')