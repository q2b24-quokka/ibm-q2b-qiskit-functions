#!/usr/bin/python3

import sys
import re
import qiskit
import math
import numpy as np
import pandas as pd
from qiskit.providers import JobStatus
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
pd.set_option('display.max_rows', None)

def clean_sv(sv):
    s1 = re.sub('\. +','.0',str(sv))
    return re.sub('(\+|-)0\.j','',s1)


def simulate_sv(circuit,decimals=3):
    sv_sim = Aer.get_backend('statevector_simulator')
    job = execute(circuit, sv_sim) 
    result = job.result() 
    outputstate = result.get_statevector(circuit, decimals=decimals)
    print(clean_sv(str(outputstate)))
    
def simulate_unitary(circuit):
    unitary_sim = Aer.get_backend('unitary_simulator')
    job = execute(circuit, unitary_sim)
    result = job.result()
    return result.get_unitary(circuit, decimals=3)
    
def simulate_measurement(circuit):
    qasm_sim = Aer.get_backend('qasm_simulator') 
    job = execute(circuit, qasm_sim, shots=1000) 
    result = job.result() 
    counts = result.get_counts(circuit)
    print(counts)

# Convert SV from matrix to equation
def print_sv_string(vector, i=0, cs=""):
    factored_sv = factor_sv(vector, i)
    sv_string = f"{cs} * (" if cs else ""
    first = True
    for idx, coeff in enumerate(factored_sv):
        if coeff == 1:
            if first:
                first = False
            else:
                sv_string += (" + ")
            sv_string += (f"|{idx:04b}> ")
        elif coeff == -1:
            if first:
                first = False
            else:
                sv_string += (" - ")
            sv_string += (f"|{idx:04b}>")
    sv_string += ")"
    return sv_string


# Get state vector with constant factored out
def factor_sv(vector, i):
    factored_sv = []
    for coeff in vector:
        if (coeff == np.complex128(0.+i)) or (coeff == np.complex128(-0.+i)):
            factored_sv.append(1)
        elif (coeff == np.complex128(0.-i)) or (coeff == np.complex128(-0.-i)):
            factored_sv.append(-1)
        else:
            factored_sv.append(0)
    return factored_sv


# Get eigensets of operator
def get_eigensets(sv_coeffs):
    esets = []
    curr_vector = np.zeros(len(sv),dtype=np.complex_)
    for idx in range(len(sv)):
        if sv_coeffs[idx] != 0:
            curr_vector[idx] = sv_coeffs[idx]
            esets.append((f"{idx:04b}", curr_vector))
            curr_vector = np.zeros(len(sv),dtype=np.complex_)
    return esets

# Calculate probability of each eigenvalue
def calc_probabilities(esets, sv_coeffs):
    probs = {}
    for eset in esets:
        inner_prod = np.vdot(eset[1], sv_coeffs)
        probs[eset[0]] = np.absolute(inner_prod)
    return probs

# Calculate mean squared error
def calc_mse(a,b):
    diffs = np.subtract(a, b) # calculate differences between true and predicted values
    squares = np.square(diffs) # find squares of differences
    return np.mean(squares) # calculate mean of squares to get score

# Generate table view of operator matrix
def factor_op(op, i):
    factored_op = []
    for vector in op:
        factored_vector = []
        for coeff in vector:
            if (coeff == np.complex128(0.+i)) or (coeff == np.complex128(-0.+i)):
                factored_vector.append(1)
            elif (coeff == np.complex128(0.-i)) or (coeff == np.complex128(-0.-i)):
                factored_vector.append(-1)
            else:
                factored_vector.append(0)
        factored_op.append(factored_vector)
    return factored_op

def print_op_matrix(op, i=0, factored=True):
    op_rows = factor_op(op, i) if factored else op
    labels = [f"{idx:04b}" for idx in range(len(op_rows[0]))]
    df = pd.DataFrame(op_rows, columns=labels)
    df.index = labels
    return df

# Generate table view of circuit instruction state vector
def sv_table(column_vector, column_label=['statevector']):
    eps = np.finfo(float).eps
    num_rows = len(column_vector)
    padding = math.ceil(np.log(num_rows)/np.log(2))
    sememe_labels = ['{v:0{p}b}'.format(v=idx, p=padding) for idx in range(num_rows)]
    
    column_vector[np.abs(column_vector) < eps] = 0

    clean_sv = []
    for val in column_vector:
        if abs(val) == 0:
            clean_sv.append('0')
        else:
            val_str = f'{val.real} + {val.imag}i'
            clean_sv.append(val_str)
    # df = pd.DataFrame(clean_sv, index=sememe_labels, columns=column_label)
    df = pd.DataFrame(clean_sv, columns=column_label)
    return df


# Generate table view of circuit instruction state vector
def nonzero_states(column_vector):
    eps = np.finfo(float).eps
    column_vector[np.abs(column_vector) < eps] = 0
    nonzero_states = set()
    for state, amp in enumerate(column_vector):
        if abs(amp) != 0:
            nonzero_states.add(state)
    return nonzero_states

def print_nonzero(column_vector):
    eps = np.finfo(float).eps
    num_rows = len(column_vector)
    padding = math.ceil(np.log(num_rows)/np.log(2))
    sememe_labels = ['{v:0{p}b}'.format(v=idx, p=padding) for idx in range(num_rows)]

    column_vector[np.abs(column_vector) < eps] = 0

    clean_sv = []
    for val in column_vector:
        if abs(val) == 0:
            clean_sv.append('0')
        else:
            val_str = f'{val.real} + {val.imag}i'
            clean_sv.append(val_str)

    for idx, val in enumerate(clean_sv):
        if val != '0':
            print(f'{sememe_labels[idx]}\t{val}')


def save_shs_data(raw_docs, corpus, shs, suffix):
    with open(f'out/shs_{suffix}.txt', 'w') as f:
        f.write('---------------------- CORPUS ----------------------\n')
        f.write('\nraw text | vectors | word/sememes\n')
        for i, doc in enumerate(corpus.docs):
            f.write(f'\n{i+1}. {raw_docs[i]}\n')
            f.write('{:<2} {:<}'.format('', str(shs.corpus[i])))
            f.write('\n')
            for entry in doc:
                sems = ', '.join(sorted(list(shs.word_to_sem[entry])))
                f.write('{:<2} {:<20} {:<2} {:<}'.format('', str(entry), '', sems))
                f.write('\n')
        
        f.write('\n\n\n------------- SEMANTIC HILBERT SPACE -------------\n')
        f.write('{:<14} {:<}'.format('Words:', len(corpus.entries)))
        f.write('\n')
        f.write('{:<14} {:<}'.format('Sememes:', shs.num_sememes))
        f.write('\n')
        f.write('{:<14} {:<}'.format('Qubits:', math.ceil(np.log(shs.num_sememes)/np.log(2))))
        f.write('\n')
        all_sems = ', '.join(list(shs.sem_to_val.keys()))
        f.write('{:<14} {:<}'.format('Sememe List:', all_sems))

def print_expected(corpus, query):
    print('{:<6} {:<}'.format('DID', 'MATCHES'))
    for idx, doc in enumerate(corpus):
        intersect = doc.intersection(query)
        if len(intersect) > 0:
            matches = ', '.join(str(m) for m in list(intersect))
            print('{:<6} {:<}'.format(idx+1, matches))

def get_decimal_counts(exp_results):
    exp_counts = []
    for res in exp_results:
        counts = res.get_counts()
        exp_counts.append(counts)
    dec_counts = []
    for c in exp_counts:
        decimal_dict = {int(k, 2):v for k,v in list(c.items())}
        dec_counts.append(decimal_dict)
    return dec_counts

def get_answers(dec_counts, shots, threshold=5):
    answers = []
    for counts in dec_counts:
        sorted_counts = {k: v for k, v in sorted(counts.items(), reverse=True, key=lambda item: item[1])}
        # doc_answers = {}
        doc_answers = []
        for candidate, count in sorted_counts.items():
            pct = round(count/shots*100, 3)
            if pct < threshold:
                break
            else:
                # doc_answers[candidate] = pct
                doc_answers.append(candidate)
        answers.append(doc_answers)
    return answers

def get_answers_top(dec_counts, shots):
    answers = []
    for counts in dec_counts:
        sorted_counts = {k: v for k, v in sorted(counts.items(), reverse=True, key=lambda item: item[1])}
        top_count = list(sorted_counts.values())[0]
        top_answers = set([ k for k,v in list(sorted_counts.items()) if v >= top_count ])
        answers.append(top_answers)
    return answers

def print_answers(answers):
    for i,a in enumerate(answers):
        print(f'DOC {i+1}:  {a}')

def plot_exp_counts(dec_counts, top_num):
    # dec_counts = get_decimal_counts(exp_results)
    for idx, c in enumerate(dec_counts):
        title = f'Document {idx+1}'
        display(plot_histogram(data=c, title=title, sort='value_desc', number_to_keep=top_num))