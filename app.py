from flask import Flask, render_template, request, jsonify
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

app = Flask(__name__)

# global circuit (1 qubit like your app)
circuit = QuantumCircuit(1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/reset", methods=["POST"])
def reset():
    global circuit
    circuit = QuantumCircuit(1)
    return jsonify({"status": "reset"})

@app.route("/gate", methods=["POST"])
def apply_gate():
    global circuit
    gate = request.json["gate"]

    if gate == "X":
        circuit.x(0)
    elif gate == "Y":
        circuit.y(0)
    elif gate == "Z":
        circuit.z(0)
    elif gate == "H":
        circuit.h(0)
    elif gate == "S":
        circuit.s(0)
    elif gate == "SD":
        circuit.sdg(0)
    elif gate == "T":
        circuit.t(0)
    elif gate == "TD":
        circuit.tdg(0)

    state = Statevector.from_instruction(circuit)
    prob_0 = float(abs(state.data[0])**2)
    prob_1 = float(abs(state.data[1])**2)

    return jsonify({
        "prob_0": round(prob_0, 4),
        "prob_1": round(prob_1, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
