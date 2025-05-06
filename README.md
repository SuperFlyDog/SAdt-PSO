🧠 SET-PSO: Self-Evaluated Topology Particle Swarm Optimization (with BBExp & PSO Comparison)
📌 Description
This repository presents a comparative study and implementation of three Particle Swarm Optimization (PSO) algorithms:

🔬 SAdt-PSO (SET-PSO): A novel PSO algorithm with dynamic topology evaluation using the Metropolis criterion.

🧪 BBExpPSO: A Bare Bones PSO variant that alternates between deterministic and Gaussian updates.

🧬 Traditional PSO: Standard PSO with fixed global topology and velocity-based update.

These implementations include experiments on high-dimensional optimization tasks, logged into test_results.csv for reproducibility and analysis.

🗂 File Structure
bash
复制
编辑
.
├── SAdtPSO.py            # SET-PSO with topology adaptation and Metropolis-based evaluation
├── BBExpPSO.py           # BBExp particle swarm optimization algorithm
├── PSO.py                # Traditional velocity-based PSO
├── test_results.csv      # Performance results from benchmark tests
├── README.md             # Project overview and instructions
└── requirements.txt      # Python dependency list
🔍 Algorithms Overview
🔁 SAdt-PSO (Self-Adaptive Topology PSO)
Adaptive topology reconnection using local distance and Metropolis acceptance.

Uses X-Von Neumann neighborhood topology.

Integrates stagnation detection via sliding window of fitness improvements (Δf_best).

Employs Bare Bones-style update rules.

📉 BBExpPSO
Particles update positions with 50% probability of using pbest or Gaussian sampling.

No explicit velocity, inertia, or acceleration coefficients needed.

Robust and parameter-free.

⚙️ Traditional PSO
Standard position and velocity updates using inertia and cognitive/social components.

Implements classic global best behavior with adjustable hyperparameters.

📈 Benchmark and Evaluation
Benchmark function used (example):

text
复制
编辑
Shifted Rotated Griewank Function:
f(x) = 1 + (1 / 4000) * Σ((x_i - O_i)^2) - Π(cos((x_i - O_i) / sqrt(i)))
Metrics captured across 10 runs:

✅ Mean Empirical Error

✅ Standard Deviation of Error

✅ Mean Execution Time

✅ Reliability Index (based on time + stability)

🧪 How to Run
🔧 Install Dependencies
bash
复制
编辑
pip install -r requirements.txt
▶️ Execute Algorithms
Each script can be run independently:

bash
复制
编辑
python SAdtPSO.py        # Run SET-PSO with adaptive topology
python BBExpPSO.py       # Run BBExp PSO
python PSO.py            # Run traditional PSO
Results will be saved in test_results.csv.

📊 Example Output Format (test_results.csv)
Algorithm	Test Name	Mean Error	Std Error	Mean Time (s)	Parameters
30-SAdt-PSO	Griewank	0.00231	0.00115	1.243	{...}
BBExpPSO	Sphere	0.00412	0.00203	0.852	{...}
Traditional PSO	Rastrigin	3.182	1.124	0.993	{...}

🧠 Design Highlights
Modular code structure (easy to extend with more test functions or swarm behaviors)

Fully reproducible experiments via n_runs loop

CSV logger for downstream analysis or plotting

Potential for parallelism or CUDA acceleration (SET-PSO)

📄 License
This project is open-source under the MIT License.
© 2025 Kang Yin — Hosei University Graduate School of Computer Science
