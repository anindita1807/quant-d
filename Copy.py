import qiskit
import qiskit_aer
import numpy as np
import tkinter
import warnings
import matplotlib.pyplot as plt  # Import Matplotlib for Bloch sphere plotting
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit.quantum_info import Statevector, DensityMatrix
from qiskit.visualization import visualize_transition, plot_bloch_vector
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from functools import partial
from qiskit.qasm3 import dump
from tkinter import simpledialog
from tkinter import LEFT, END, DISABLED, NORMAL
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

#Defining Window
root = tkinter.Tk()
root.title("Quant-D")
root.iconbitmap(default = 'logo.ico')#Setting the icon

root.geometry('399x650')
#root.geometry('500x700')
root.resizable(0,0) #Blocking the resizabe feature

#Define the colors and fonts
background = '#0a1109'
buttons = '#834558'
special_buttons = '#bc3454'
button_font = ('Times New Roman', 18)
display_font = ('Times New Roman', 32,)

# Define Functions
def about():
    """
    Displays Info about the application
    """
    info = tkinter.Tk()
    info.title('About')
    info.geometry('650x470')
    info.resizable(0,0)

    text = tkinter.Text(info, height=20, width=20)

    #Creating Label
    label = tkinter.Label(info, text = "About Quant-D: ") # Corrected tkinter.label to tkinter.Label
    label.config(font=("Arial", 14))

    text_to_display = """ """

    label.pack()
    text.pack(fill='both', expand = True)

    #Insert text
    text.insert(END, text_to_display)

    #run
    info.mainloop()

# Initialize the Quantum Circuit
circuit = None  # Initialize circuit outside the function if you intend to use it globally
def initialize_circuit():
    global circuit
    circuit = QuantumCircuit(1)
    return circuit # Return the circuit if you're not strictly relying on the global variable

initialize_circuit()
theta = 0

gate_sequence = []
step_index = [0]  # using a list to make it mutable

def display_gate(gate_input):
    display.insert(END, gate_input)
    input_gates = display.get()
    num_gates_pressed = len(input_gates)
    list_input_gates = list(input_gates)
    search_word = ["R", "D"]
    count_double_valued_gates = [list_input_gates.count(i) for i in search_word]
    num_gates_pressed -= sum(count_double_valued_gates)
    if num_gates_pressed == 10:
        gates = [x_gate, y_gate, z_gate, Rx_gate, Ry_gate, Rz_gate, s_gate, sd_gate, t_gate, td_gate, hadamard]
        for gate in gates:
            gate.config(state=DISABLED)

    # Append the gate operation to the gate_sequence
    if gate_input == "X":
        gate_sequence.append(lambda: circuit.x(0))
    elif gate_input == "Y":
        gate_sequence.append(lambda: circuit.y(0))
    elif gate_input == "Z":
        gate_sequence.append(lambda: circuit.z(0))
    elif gate_input == "H":
        gate_sequence.append(lambda: circuit.h(0))
    elif gate_input == "S":
        gate_sequence.append(lambda: circuit.s(0))
    elif gate_input == "SD":
        gate_sequence.append(lambda: circuit.sdg(0))
    elif gate_input == "T":
        gate_sequence.append(lambda: circuit.t(0))
    elif gate_input == "TD":
        gate_sequence.append(lambda: circuit.tdg(0))

    history_box.config(state=NORMAL)
    history_box.insert(END, gate_input + "\n")
    history_box.config(state=DISABLED)

'''
def step_replay():
    initialize_circuit()  # always reset before replaying
    for i in range(step_index[0] + 1):
        gate_sequence[i]()  # apply gates up to this step
    update_state_display()
    step_index[0] += 1
    if step_index[0] >= len(gate_sequence):
        step_button.config(state=DISABLED)
'''

def show_qasm():
    initialize_circuit()
    for gate in gate_sequence:
        gate()

    qasm_window = tkinter.Toplevel(root)
    qasm_window.title("QASM Code")
    qasm_window.geometry("600x400")
    qasm_window.resizable(0, 0)
    qasm_text = tkinter.Text(qasm_window, wrap='word', font=("Courier", 10), bg="#1b1b1b", fg="#20ff00")
    qasm_text.pack(fill='both', expand=True, padx=10, pady=10)

    try:
        from io import StringIO
        from qiskit import transpile

        # Transpile to preserve the gates you want in the output
        # Transpilation is initialized but not used since rx, ry, and rz is future scope in this QASM code
        basis = ['s', 't', 'sdg', 'tdg', 'h']
        transpiled_circuit = transpile(circuit, basis_gates=basis)

        buffer = StringIO()
        dump(circuit, buffer)  # This exports QASM 3
        qasm_code = buffer.getvalue()

        if not qasm_code.strip():
            qasm_code = "No QASM code available. Is the circuit empty?"

    except Exception as e:
        qasm_code = f"Error generating QASM:\n{str(e)}"

    qasm_text.insert(END, qasm_code)
    qasm_text.config(state=DISABLED)

# Function to apply noise to the quantum circuit based on user input and visualize the noisy state
def apply_noise_and_visualize(circuit, noise_type, error_rate):
    noise_model = NoiseModel()

    if noise_type == 'Depolarizing':
        error = depolarizing_error(error_rate, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u', 'rx', 'ry', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z'])
        print(noise_model)
    elif noise_type == 'Amplitude Damping':
        error = amplitude_damping_error(error_rate)
        noise_model.add_all_qubit_quantum_error(error, ['u', 'rx', 'ry', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z'])
        print(noise_model)
    elif noise_type == 'Phase Damping':
        error = phase_damping_error(error_rate)
        noise_model.add_all_qubit_quantum_error(error, ['u', 'rx', 'ry', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z'])
        print(noise_model)

    # Apply the noise model to the quantum circuit
    simulator = qiskit_aer.AerSimulator(noise_model=noise_model)

    # Circuit for density matrix simulation *with noise*
    noisy_circuit_densitymatrix = circuit.copy()
    print("noisy_circuit_densitymatrix: ", noisy_circuit_densitymatrix)
    noisy_circuit_densitymatrix.save_density_matrix() # Save density matrix
    noisy_transpiled_densitymatrix = transpile(noisy_circuit_densitymatrix, simulator)
    print("noisy_transpiled_densitymatrix: ", noisy_transpiled_densitymatrix)
    noisy_result_densitymatrix = simulator.run(noisy_transpiled_densitymatrix).result()
    print("noisy_result_densitymatrix: ", noisy_result_densitymatrix)

    # Get the noisy density matrix
    try:
        experiment_result = noisy_result_densitymatrix.results[0]
        print("experiment_result: ", experiment_result)
        density_matrix_data = experiment_result.data.density_matrix
        print("density_matrix_data: ", density_matrix_data)

        if density_matrix_data is not None:
            noisy_densitymatrix = DensityMatrix(density_matrix_data)
            print(f"Noisy Density Matrix:\n{noisy_densitymatrix.data}")
            rho = noisy_densitymatrix.data
            print("rho: ", rho)
            bloch_vector = [2 * np.real(rho[0, 1]),
                            2 * np.imag(rho[0, 1]),
                            rho[0, 0] - rho[1, 1]]
            print(f"Calculated Bloch Vector (Noisy DensityMatrix): {bloch_vector}")
        else:
            print("Error: 'density_matrix' not found in the result data.")
            bloch_vector = [0.0, 0.0, 0.0]

    except Exception as e:
        print(f"Error getting noisy density matrix: {e}")
        bloch_vector = [0.0, 0.0, 0.0] # Default if error

    # Circuit for counting (with measurement and noise)
    measured_noisy_circuit = transpile(circuit.copy(), simulator)
    measured_noisy_circuit.measure_all()
    result_counts = simulator.run(measured_noisy_circuit, shots=1024).result()
    counts = result_counts.get_counts()
    total_shots = sum(counts.values())
    prob_0 = counts.get('0', 0) / total_shots
    prob_1 = counts.get('1', 0) / total_shots
    prob_label.config(text=f"Applied {noise_type} noise with error rate {error_rate}\nProbability of |0⟩: {prob_0:.2f}\nProbability of |1⟩: {prob_1:.2f}", font=("Courier", 9))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_bloch_vector(bloch_vector, ax=ax)
    ax.set_title(f"Bloch Sphere with {noise_type} (Error Rate: {error_rate:.2f})")
    plt.show()


# Callback function when the "Noise" button is clicked
def noise_button_clicked():
    # Create a new window to select noise type and error rate
    noise_window = tkinter.Toplevel(root)
    noise_window.title("Configure Noise")

    # Add a label for noise type selection
    tkinter.Label(noise_window, text="Select Noise Type:").pack(pady=5)

    # Create a list of noise types
    noise_types = ['Depolarizing', 'Amplitude Damping', 'Phase Damping']
    noise_type_var = tkinter.StringVar()
    noise_type_var.set(noise_types[0])  # Set default to Depolarizing

    noise_type_menu = tkinter.OptionMenu(noise_window, noise_type_var, *noise_types)
    noise_type_menu.pack(pady=10)

    # Add a label and input field for the error rate
    tkinter.Label(noise_window, text="Enter Error Rate (0 to 1):").pack(pady=5)
    error_rate_entry = tkinter.Entry(noise_window)
    error_rate_entry.pack(pady=10)
    error_rate_entry.insert(0, "0.05")  # Default value for error rate

    # Apply button callback
    def apply_noise_config():
        try:
            error_rate = float(error_rate_entry.get())
            if not 0 <= error_rate <= 1:
                tkinter.messagebox.showerror("Error", "Error rate must be between 0 and 1.")
                return
            noise_type = noise_type_var.get()

            # Create a copy of the current circuit (important to not modify the original)
            temp_circuit = circuit.copy()

            # Apply noise and visualize
            apply_noise_and_visualize(temp_circuit, noise_type, error_rate)

            noise_window.destroy()  # Close the noise configuration window

        except ValueError:
            tkinter.messagebox.showerror("Error", "Invalid error rate. Please enter a number.")

    # Add an "Apply" button to apply the selected noise and error rate
    apply_button = tkinter.Button(noise_window, text="Apply Noise & Visualize", command=apply_noise_config) # Changed button text
    apply_button.pack(pady=20)

def create_noise_model(noise_type, error_rate):
    noise_model = NoiseModel()

    if noise_type == 'Depolarizing':
        error = depolarizing_error(error_rate, 1)
        noise_model.add_all_qubit_quantum_error(error, ['u', 'rx', 'ry', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z']) # Apply to more common gates
    elif noise_type == 'Amplitude Damping':
        error = amplitude_damping_error(error_rate)
        noise_model.add_all_qubit_quantum_error(error, ['u', 'rx', 'ry', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z'])
    elif noise_type == 'Phase Damping':
        error = phase_damping_error(error_rate)
        noise_model.add_all_qubit_quantum_error(error, ['u', 'rx', 'ry', 'rz', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z'])

    return noise_model

def clear(circuit):
    display.delete(0, END)
    initialize_circuit()

    if x_gate['state'] == DISABLED:
        gates = [x_gate, y_gate, z_gate, Rx_gate, Ry_gate, Rz_gate, s_gate, sd_gate, t_gate, td_gate, hadamard]
        for gate in gates:
            gate.config(state=NORMAL)

    # Clear state display and probabilities
    state_label.config(text="")
    prob_label.config(text="Probability of |0⟩: 1.00\nProbability of |1⟩: 0.00")


    gate_sequence.clear()
    step_index[0] = 0
    #step_button.config(state=NORMAL)


def visualize_circuit(circuit, window):
    """
    Visualizes the single qubit rotations corresponding to applied gates in a separate tkinter window
    Handles any possible visualization error

    """
    try:
        visualize_transition(circuit=circuit)
    except qiskit.visualization.exceptions.VisualizationError:
        window.destroy()

# Changing the value of theta: Changes global value of theta and destroys the window
def change_theta(num, window, circuit, key):
    global theta
    theta = num * np.pi

    # Apply the gate
    if key == 'x':
        circuit.rx(theta, 0)
        gate_sequence.append(partial(circuit.rx, theta, 0))
    elif key == 'y':
        circuit.ry(theta, 0)
        gate_sequence.append(partial(circuit.ry, theta, 0))
    elif key == 'z':
        circuit.rz(theta, 0)
        gate_sequence.append(partial(circuit.rz, theta, 0))

    update_state_display()
    theta = 0
    window.destroy()


# User Input for parameterized Rotation gates, Rx Ry Rz
def update_state_display():
    state = Statevector.from_instruction(circuit)
    prob_0 = abs(state.data[0])**2
    prob_1 = abs(state.data[1])**2

    # Format probability text
    prob_text = f"Probability of |0⟩: {prob_0:.2f}\nProbability of |1⟩: {prob_1:.2f}"
    prob_label.config(text=prob_text, font=("Courier", 10))


def user_input(circuit, key):

    #Initialize and define the properties of window
    get_input = tkinter.Tk()
    get_input.title('Get Theta')
    get_input.geometry('360x160')
    get_input.resizable(0,0)

    val1 = tkinter.Button(get_input, height=2, width=10, bg='black', fg = '#20ff00',font= ("Arial, 10"), text='PI/4', command=lambda:change_theta(0.25, get_input, circuit, key))
    val1.grid(row=0, column=0)

    val2 = tkinter.Button(get_input, height=2, width=10, bg='black', fg = '#20ff00',font= ("Arial, 10"), text='PI/2', command=lambda:change_theta(0.50, get_input, circuit, key))
    val2.grid(row=0, column=1)

    val3 = tkinter.Button(get_input, height=2, width=10, bg='black', fg = '#20ff00',font= ("Arial, 10"), text='PI', command=lambda:change_theta(1.0, get_input, circuit, key))
    val3.grid(row=0, column=2)
    val4 = tkinter.Button(get_input, height=2, width=10, bg='black', fg = '#20ff00',font= ("Arial, 10"), text='2*PI', command=lambda:change_theta(2.0, get_input, circuit, key))
    val4.grid(row=0, column=3, sticky = 'W')

    nval1 = tkinter.Button(get_input, height=2, width=10, bg='black', fg = '#20ff00',font= ("Arial, 10"), text='-PI/4', command=lambda:change_theta(-0.25, get_input, circuit, key))
    nval1.grid(row=1, column=0)

    nval2 = tkinter.Button(get_input, height=2, width=10, bg='black', fg = '#20ff00',font= ("Arial, 10"), text='-PI/2', command=lambda:change_theta(-0.50, get_input, circuit, key))
    nval2.grid(row=1, column=1)

    nval3 = tkinter.Button(get_input, height=2, width=10, bg='black', fg = '#20ff00',font= ("Arial, 10"), text='-PI', command=lambda:change_theta(-1.0, get_input, circuit, key))
    nval3.grid(row=1, column=2)

    nval4 = tkinter.Button(get_input, height=2, width=10, bg='black', fg = '#20ff00',font= ("Arial, 10"), text='-2*PI', command=lambda:change_theta(-2.0, get_input, circuit, key))
    nval4.grid(row=1, column=3, sticky = 'W')

    text_object = tkinter.Text(get_input, height = 20, width = 20, bg= "light grey")

    note = """
    GIVE THE VALUE FOR THE THETA
    The value has the range [-2*PI, 2*PI]
    """

    text_object.grid(sticky='WE', columnspan=4)
    text_object.insert(END, note)

    get_input.mainloop()



#define Layout
#define Frames
display_frame = tkinter.LabelFrame(root)
button_frame = tkinter.LabelFrame(root, bg = 'black')
display_frame.pack()
button_frame.pack(fill='both', expand = True)

# Define the Display Frame Layout
display = tkinter.Entry(display_frame, width=120, font=display_font, bg=background, fg = '#20ff00', borderwidth=10, justify=tkinter.LEFT)
display.pack(padx=3,pady=4)
state_label = tkinter.Label(root, text="", fg="#20ff00", bg="#0a1109", font=("Courier", 12))
state_label.pack(pady=10)

# First row
x_gate = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = '#20ff00', text = 'X', command = lambda:[display_gate('X'), circuit.x(0), update_state_display()])
y_gate = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = '#20ff00', text = 'Y', command = lambda:[display_gate('Y'), circuit.y(0), update_state_display()])
z_gate = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = '#20ff00', text = 'Z', command = lambda:[display_gate('Z'), circuit.z(0), update_state_display()])
x_gate.grid(row=0, column=0, ipadx = 45, pady = 1)
y_gate.grid(row=0, column=1, ipadx = 45, pady = 1)
z_gate.grid(row=0, column=2, ipadx = 53, pady = 1, sticky = 'E')

# Define the Second Row of buttons
Rx_gate = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = 'red', text = 'RX', command=lambda:[display_gate('Rx'), user_input(circuit,'x')])
Ry_gate = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = 'red', text = 'RY', command=lambda:[display_gate('Ry'), user_input(circuit,'y')])
Rz_gate = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = 'red', text = 'RZ', command=lambda:[display_gate('Rz'), user_input(circuit,'z')])
Rx_gate.grid(row=1, column=0,columnspan=1,sticky='WE', pady=1)
Ry_gate.grid(row=1, column=1,columnspan=1,sticky='WE', pady=1)
Rz_gate.grid(row=1, column=2,columnspan=1,sticky='WE', pady=1)


# Define the third row of buttons
s_gate = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = '#20ff00', text = 'S', command = lambda:[display_gate('S'), circuit.s(0), update_state_display()])
sd_gate = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = '#20ff00', text = 'SD', command = lambda:[display_gate('SD'), circuit.sdg(0), update_state_display()])
hadamard = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = '#20ff00', text = 'H', command = lambda:[display_gate('H'), circuit.h(0), update_state_display()])
s_gate.grid(row=2, column=0,columnspan=1,sticky='WE', pady=1)
sd_gate.grid(row=2, column=1,sticky='WE', pady=1)
hadamard.grid(row=2, column=2,rowspan=2,sticky='WENS', pady=1)

# Define fifth row of buttons
t_gate = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = '#20ff00', text = 'T', command = lambda:[display_gate('T'), circuit.t(0)])
td_gate = tkinter.Button(button_frame, font = button_font, bg = 'black', fg = '#20ff00', text = 'TD', command = lambda:[display_gate('TD'), circuit.tdg(0)])
t_gate.grid(row=3, column=0,sticky='WE', pady=1)
td_gate.grid(row=3, column=1,sticky='WE', pady=1)

# Displaying the State Vector Probablities
probability_label = tkinter.Label(button_frame, text="Qubit Probabilities", bg="#0a1109", fg="#20ff00", font=("Courier", 10), anchor='w')
probability_label.grid(row=4, column=0, columnspan=3, sticky='WE')

prob_label = tkinter.Label(button_frame, text="Probability of |0⟩: 1.00\nProbability of |1⟩: 0.00", bg="#1b1b1b", fg="#20ff00", font=("Courier", 12), anchor='w', justify=LEFT)
prob_label.grid(row=5, column=0, columnspan=3, sticky='WE', padx=4, pady=2)

# History Label
history_label = tkinter.Label(button_frame, text="Gate History", bg="#0a1109", fg="#20ff00", font=("Courier", 10), anchor='w')
history_label.grid(row=6, column=0, columnspan=3, sticky='WE', pady=(6, 0))

# History Display (Scrollable Text Box)
history_box = tkinter.Text(button_frame, height=4, width=40, bg="#1b1b1b", fg="#20ff00", font=("Courier", 11))
history_box.grid(row=7, column=0, columnspan=3, sticky='WE', padx=4)
history_box.config(state=DISABLED)

# Define the Quit and Visualize button
quit = tkinter.Button(button_frame, font = button_font, bg = '#2b2b2b', fg = '#20ff00', text = 'Quit', command=root.destroy)
visualize = tkinter.Button(button_frame, font = button_font, bg = '#2b2b2b', fg = '#20ff00', text = 'Visualize', command=lambda:visualize_circuit(circuit, root))
quit.grid(row=8, column=0,columnspan=2,sticky='WE',ipadx=5, pady=1)
visualize.grid(row=8, column=2,columnspan=2,sticky='WE',ipadx=8, pady=1)

# QASM and Noise buttons in one row
qasm_button = tkinter.Button(button_frame, text="QASM", font=button_font, bg="#2b2b2b", fg="#20ff00", command=lambda: show_qasm())
noise_button = tkinter.Button(button_frame, text="Noise", font=button_font, bg="#2b2b2b", fg="#20ff00", command=lambda: noise_button_clicked())
qasm_button.grid(row=9, column=0, columnspan=2, sticky='WE', padx=1, pady=1)
noise_button.grid(row=9, column=2, columnspan=2, sticky='WE', padx=1, pady=1)

# Define Step Button
#step_button = tkinter.Button(button_frame, font=button_font, bg='#2b2b2b', fg='#20ff00', text="Step ▶", command=step_replay)
#step_button.grid(row=10, column=0, columnspan=3, sticky='WE')

# Define the clear button
clear_button = tkinter.Button(button_frame, font = button_font, bg = '#2b2b2b', fg = '#20ff00', text = 'Clear', command=lambda:clear(circuit))
clear_button.grid(row=11, column=0, columnspan=3, sticky='WE')

# Define the about button
about_button = tkinter.Button(button_frame, font = button_font, bg = '#2b2b2b', fg = '#20ff00', text = 'About', command=lambda:about())
about_button.grid(row=12,column=0,columnspan=3,sticky='WE')

# Main Loop
root.mainloop()
