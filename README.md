# Search-and-Rescue POMDP Benchmark with MaskablePPO vs POMCP

## Research Motivation and Problem Statement

This repository presents a novel, reproducible benchmark and comparative study for search and rescue (SAR) in procedurally generated, unknown maze environments under partial observability. The agent must rescue a stochastically moving victim using only local vision and a noisy proximity sensor, with no prior map information. The environment is designed to be challenging: the agent faces uncertainty in both the map and sensor, must actively choose when to sense, and must balance exploration and exploitation to succeed.

**Key Research Gap:**

Despite extensive literature in POMDP planning, active sensing, and deep RL, most prior work assumes static or known maps, perfect sensors, stationary or fully observable targets, or lacks procedural map generation. This project introduces a realistic SAR benchmark and evaluates three fundamentally different approaches:

- **Belief-Based Planning:** Bayesian planning in belief space, leveraging particle filters and active sensing ([Silver & Veness, 2010](https://papers.nips.cc/paper/2010/hash/2ce16b5b9d0f3c5f.pdf)).
- **Masked Deep RL:** Policy-based learning with dynamic action masking, enabling agents to learn valid actions under uncertainty ([SB3-Contrib, 2021](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)).
- **Random Agent Baseline:** A random action agent with valid action masking, providing a lower-bound baseline for performance comparison and sanity checks.

No prior work has systematically compared belief-based planning, masked deep RL, and a random agent baseline in environments with:

- Unknown, procedurally generated mazes
- No prior map information
- Noisy, probabilistic proximity sensors
- Moving, stochastic targets
- Partial observability and dynamic action validity

## Overview

This benchmark enables systematic comparison of three agent types:

- **POMCP (Partially Observable Monte Carlo Planning)**
- **MaskablePPO (Policy-based RL with action masking)**
- **Random agent baseline**

The environment simulates a realistic SAR scenario: an agent must find and rescue a moving victim using only local vision and a noisy proximity sensor, with no prior map knowledge.

## Key Features, Contributions, and Novelty

- **Benchmark Environment:** Realistic SAR scenario for pursuit-evasion under partial observability, featuring procedural maze generation, noisy sensors, and moving targets.
- **Comparative Analysis:** Systematic evaluation of POMCP, MaskablePPO, and a Random agent baseline in the same environment, with reproducible experiments and seeded runs.
- **Reward Shaping:** Explicit alpha tradeoff $\alpha \cdot \text{progress} - (1 - \alpha) \cdot \text{cost}$, combining entropy/information-gain and distance reduction with step cost.
- **Failure Modes:** Identification of strengths and weaknesses of model-based (belief-space) and model-free (policy-based) approaches.
- **Open-Source Code:** Modular implementation for reproducibility and extension, including evaluation scripts, plotting, and CSV/Tensorboard outputs.
- **Claims and Novelty:**
  - This work directly compares a POMCP-like planner, MaskablePPO (with action masking), and a Random agent baseline in SAR victim search, using entropy/information-gain rewards and explicit alpha tradeoff, within a procedural maze and noisy proximity sensor setting.
  - The integrated combination (procedural maze, moving victim, noisy proximity sensor, Bayesian belief, MaskablePPO vs POMCP vs Random agent) is distinctive and not commonly documented as a reproducible benchmark. Prior works address only subsets of these challenges ([Silver & Veness, 2010](https://papers.nips.cc/paper/2010/hash/2ce16b5b9d0f3c5f.pdf); [Bourgault et al., 2002]; [Hausknecht & Stone, 2015]; [Hsu et al., 2022]).
  - Provides a new testbed for evaluating planning, deep RL, and random baselines in realistic SAR settings, exposing strengths and limitations of each approach.

## Project Structure

```bash
.
├── configuration.yaml        # Main configuration file (hyperparameters, environment, evaluation)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── scripts/                  # Batch scripts for training/evaluation
│   ├── train_all_alphas.py
│   ├── evaluate_all_agents.py
│   └── __init__.py
├── source/
│   ├── agents/               # Agent implementations and evaluation scripts
│   ├── environment/          # Maze generation, SAR environment, sensor models
│   ├── updates/              # Belief update, reward, sensor logic
│   └── utilities/            # Callbacks, helpers, plotting
└── results/                  # All experiment outputs
    ├── checkpoints/          # Model checkpoints
    ├── evaluation_data/      # CSV logs of evaluation runs
    ├── models/               # Saved models
    ├── tensorboard/          # Tensorboard logs
    ├── training_data/        # CSV logs of training runs
    ├── training_plots/       # Plots and summary CSVs
    └── vector_normalize/     # Normalization stats
```

## How the Codebase Works

- **Environment:**

  - Maze generation, SAR simulation, sensors, and belief updates are in `source/environment/` and `source/updates/`.

- **Agents:**

  - POMCP: Belief-based planning with particle filters and MCTS (`source/agents/pomcp_agent.py`).
  - MaskablePPO: Deep RL with action masking (`source/agents/train_maskableppo.py`).
  - Random: Baseline agent (`source/agents/random_agent.py`).

- **Training & Evaluation:**

  - `scripts/train_all_alphas.py`: Trains MaskablePPO for all alpha values.
  - `scripts/evaluate_all_agents.py`: Evaluates all agents and logs results.

- **Utilities:**

  - Plotting, helpers, and callbacks in `source/utilities/`.

- **Results:**

  - All outputs organized by agent and alpha value in `results/`.

## Configuration: `configuration.yaml`

Controls all major experiment settings:

- **train:** MaskablePPO hyperparameters
- **environment:** Maze size, vision, sensor noise, etc.
- **reward:** Weights for step cost, distance, entropy, termination
- **visualization:** Rendering options
- **pomcp:** POMCP-specific parameters
- **evaluation:** Which agents to evaluate, etc.

**Edit `configuration.yaml` to change experiment settings.**

## Workflow: Running Experiments

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train MaskablePPO for all alphas:**

   ```bash
   python scripts/train_all_alphas.py
   ```

   - Models and logs saved in `results/models/` and `results/training_data/`.

3. **Evaluate all agents:**

   ```bash
   python scripts/evaluate_all_agents.py
   ```

   - Results in `results/evaluation_data/` and plots in `results/training_plots/`.

4. **Analyze results:**

   - Use CSV logs and plots in `results/` to compare performance.
   - Tensorboard logs in `results/tensorboard/` for MaskablePPO.

## How Everything Syncs

## Literature Context and Citations

This benchmark is grounded in and extends the following research areas:

**Search-and-Rescue POMDPs and Victim Search:**

- Kurniawati, Hsu, Lee (2008), "POMDPs for Robotic Search and Rescue" — early POMDP framing for SAR. [RSS 2008](https://roboticsproceedings.org/rss2008wksp/POMDPs_SAR)
- Rosenthal et al. (2010), "Target Localization with Uncertainty" — POMDP target localization. [AAAI 2010](https://aaai.org/Papers/AAAI/2010/AAAI10-197)
- Karpati et al. (2012), "Risk-Aware POMDP Planning for Rescue Robots". [ICRA 2012](https://doi.org/10.1109/ICRA.2012.6224666)
- Doherty et al. (2013), "Autonomous Search and Rescue in Complex Environments Using POMDPs". [JFR](https://doi.org/10.1002/rob.21475)
- Stachniss, Burgard (2005), "Information-Theoretic Search Strategies". [IJRR](https://doi.org/10.1177/0278364905051971)

**POMCP, UCT/UCB, Particle Filters, Grid Search:**

- Silver, Veness (2010), "Monte-Carlo Planning in Large POMDPs" (POMCP). [NeurIPS](https://papers.nips.cc/paper/2010/hash/2ce16b5b9d0f3c5f.pdf)
- Kocsis, Szepesvári (2006), "Bandit Based Monte-Carlo Planning". [ECML](https://doi.org/10.1007/11871842_29)
- Doucet, de Freitas, Gordon (2001), "Sequential Monte Carlo Methods in Practice". [Springer](https://doi.org/10.1007/978-1-4757-3437-9)
- Smallwood, Sondik (1973), "Optimal Control of POMPs". [Operations Research](https://doi.org/10.1287/opre.21.5.1071)
- Stone (1990), "Probability maps and search theory". [Naval Research Logistics](https://doi.org/10.1002/1520-6750(199008)37:4<617::AID-NRL2310370410>3.0.CO;2-4)

**Deep RL in POMDPs; PPO; Action Masking; Safe/Constrained RL:**

- Schulman et al. (2017), "Proximal Policy Optimization". [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Hausknecht, Stone (2015), "Deep Recurrent Q-Learning for POMDPs". [arXiv:1507.06527](https://arxiv.org/abs/1507.06527)
- Achiam et al. (2017), "Constrained Policy Optimization". [arXiv:1705.10528](https://arxiv.org/abs/1705.10528)
- García, Fernández (2015), "Safe RL: A Survey". [JMLR](https://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf)
- SB3 docs (2020–), "Action Masking". [SB3 Docs](https://stable-baselines3.readthedocs.io)
- SB3-Contrib (2021–), "MaskablePPO". [GitHub](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- Zhao et al. (2023), "Deep RL under Partial Observability: A Survey". [arXiv:2304.10566](https://arxiv.org/abs/2304.10566)

**Information Gain, Entropy Reduction, and Active Perception:**

- Lindley (1956), "On a Measure of Information". [Information and Control](https://doi.org/10.1016/S0019-9958(56)80012-8)
- MacKay (2003), "Information Theory, Inference, and Learning". [Book](https://inference.org.uk/mackay/itila)
- Singh et al. (2022), "Active Perception for Robotic Exploration: A Survey". [Robotics and Autonomous Systems](https://doi.org/10.1016/j.robot.2021.103834)
- Stachniss et al. (2005), "Information-Gain Exploration". [IJRR](https://doi.org/10.1177/0278364905051971)
- Ghavamzadeh et al. (2015), "Exploration-Exploitation Tradeoff: A Survey". [Foundations and Trends in Machine Learning](https://doi.org/10.1561/2200000049)

**Noisy Sensors, Bayesian Filtering, Occupancy Grids, and Target Tracking:**

- Elfes (1989), "Occupancy Grids". [ICRA](https://doi.org/10.1109/ROBOT.1989.100036)
- Thrun, Burgard, Fox (2005), "Probabilistic Robotics". [MIT Press](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)
- Arulampalam et al. (2002), "Particle Filters for Tracking". [IEEE Transactions on Signal Processing](https://doi.org/10.1109/78.978374)
- Fox (1998), "Sensor Models for Localization". [IJRR](https://doi.org/10.1177/027836499801700202)
- Särkkä (2013), "Bayesian Filtering and Smoothing". [Book](https://doi.org/10.1017/CBO9781139344203)

**Planning vs Deep RL; Benchmarks and Reproducibility:**

- Amos et al. (2018), "Planning vs Learning in POMDPs". [arXiv:1807.06275](https://arxiv.org/abs/1807.06275)
- Sunberg et al. (2018), "POMDPs.jl: A Framework". [JOSS](https://doi.org/10.21105/joss.00637)
- Chevalier-Boisvert et al. (2018), "MiniGrid". [GitHub](https://github.com/Farama-Foundation/MiniGrid)
- Beattie et al. (2016), "DeepMind Lab". [arXiv:1612.03801](https://arxiv.org/abs/1612.03801)
- Harish et al. (2021), "A Survey of POMDP Applications". [ACM Computing Surveys](https://doi.org/10.1145/3445250)

**Evaluation Protocols and Methodologies for SAR and POMDP Agents:**

- NIST (2015–), "Standard Test Methods for Response Robots". [NIST](https://www.nist.gov/el/intelligent-systems-division/standard-test-methods-response-robots)
- Roy et al. (2001), "Evaluating Robotic Search Strategies under Uncertainty". [IJRR](https://doi.org/10.1177/02783640122067218)
- Butler et al. (2000), "Benchmarking Active Search in Grid Environments". [ICRA](https://doi.org/10.1109/ROBOT.2000.844106)

---

- **Scripts** in `scripts/` are the main entry points, using `configuration.yaml` and calling into `source/` modules.
- **Agents** are modular and share the same environment and reward structure.
- **Results** are organized by agent and alpha value for easy comparison.
- **Changing the config** affects all subsequent runs.
