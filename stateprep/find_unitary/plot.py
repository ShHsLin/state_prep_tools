import numpy as np
import matplotlib.pyplot as plt

for seq in [3, 4]:
    for gate_type in ['real', 'complex']:
        min_list = []
        max_list = []
        num_gates = []
        for depth in range(2, 8):
            try:
                # Load the data
                data = np.loadtxt(f'sequence_{seq}_{gate_type}_depth{depth}.txt')

                min_list.append(np.min(data))
                max_list.append(np.max(data))
                num_gates.append(depth * 4 * seq)
            except:
                break

        if len(min_list) == 0:
            continue

        middle_array = (np.array(min_list) + np.array(max_list)) / 2
        # Plot the data
        plt.errorbar(num_gates, middle_array, yerr=[middle_array - np.array(min_list), np.array(max_list) - middle_array], fmt='o', capsize=5, label=f'seq{seq}+{gate_type}')


plt.xlabel('Number of gates')
plt.yscale('log')
plt.ylabel('error in fidelity')
plt.legend()
plt.show()


