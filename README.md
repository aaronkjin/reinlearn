# ReinLearn

A batch RL approach to identify deterministic policies across MDPs using Q-learning. A CS 238 project.

## Getting Started

```bash
# clone the repo
git clone https://github.com/aaronkjin/reinlearn.git
cd reinlearn

# create a virtual env
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

To run the script:

```bash
python3 q.py
```

## Technologies Used

- Python
- NumPy
- Pandas

## Background

In this project, I implement batch reinforcement learning to find optimal policies for three different Markov Decision Processes (MDPs). The implementation uses Q-learning with various optimizations for each specific problem: a small 10x10 grid world with 4 actions; the MountainCarContinuous-v0 environment with discretized states; a large-scale MDP with hidden structure.

For the MountainCar environment, I developed a specialized approach that handles the non-Markovian nature of discretized states using physics-based heuristics and reward shaping. The implementation efficiently processes both successful and unsuccessful episodes, with optimizations for handling the continuous state space after discretization.

## Developer

Aaron Jin  
[GitHub Profile](https://github.com/aaronkjin)
