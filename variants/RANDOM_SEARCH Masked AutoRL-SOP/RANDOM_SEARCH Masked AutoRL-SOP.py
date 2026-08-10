"""Masked AutoRL-SOP using Random Search as the HPO baseline."""

import tsplib95
import numpy as np
from tqdm import tqdm
import random
import time
import optuna
import os
import sys

# GENERAL CONFIGURATION
SEED_VALUE = 42
N_TRAIN_EPISODES = 10_000 
OPTUNA_TIMEOUT = 1200 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BASE_PATH = os.path.join(PROJECT_ROOT, "SOP_Datasets")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "Results")
TRAIN_PLOTS_PATH = os.path.join(SCRIPT_DIR, "Train Plots")

SOP_INSTANCES = [
    "br17.10.sop", "br17.12.sop",
    "ESC07.sop", "ESC12.sop", "ESC25.sop", "ESC47.sop", "ESC63.sop", "ESC78.sop",
    "ft53.1.sop", "ft53.2.sop", "ft53.3.sop", "ft53.4.sop",
    "ft70.1.sop", "ft70.2.sop", "ft70.3.sop", "ft70.4.sop",
    "kro124p.1.sop", "kro124p.2.sop", "kro124p.3.sop",
    "p43.1.sop", "p43.2.sop", "p43.3.sop", "p43.4.sop",
    "prob.42.sop",
    "ry48p.1.sop", "ry48p.2.sop", "ry48p.3.sop", "ry48p.4.sop"
]

BEST_KNOWN_SOLUTIONS = {
    'br17.10': 55, 'br17.12': 55,
    'ESC07': 2125, 'ESC12': 1675, 'ESC25': 1681, 'ESC47': 1288, 'ESC63': 62, 'ESC78': 18230,
    'ft53.1': 7531, 'ft53.2': 8026, 'ft53.3': 10262, 'ft53.4': 14425,
    'ft70.1': 39313, 'ft70.2': 40101, 'ft70.3': 42535, 'ft70.4': 53530,
    'kro124p.1': 38762, 'kro124p.2': 39841, 'kro124p.3': 43904,
    'p43.1': 28140, 'p43.2': 28480, 'p43.3': 28835, 'p43.4': 83005,
    'prob.42': 243,
    'ry48p.1': 15805, 'ry48p.2': 16074, 'ry48p.3': 19490, 'ry48p.4': 31446
}

class SOPEnv:
    """Tabular SARSA environment for precedence-constrained SOP instances."""
    def __init__(self, distance_matrix, precedences, epsilon, alpha, gamma, n_training_eps, problem_name="SOP", target_solution=0):
        """Initialize the environment state and SARSA parameters."""
        self.distance_matrix = distance_matrix
        self.precedences_list = precedences
        self.n_cities = distance_matrix.shape[0]
        self.target_solution = target_solution 
        
        # Precompute precedence dependencies
        # Map each visited node to the nodes made available by that visit.
        self.unlocks = [[] for _ in range(self.n_cities)]
        self.base_precedence_count = np.zeros(self.n_cities, dtype=int)
        
        for a, b in self.precedences_list:
            self.unlocks[a].append(b)
            self.base_precedence_count[b] += 1
            
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_training = n_training_eps
        self.problem_name = problem_name
        
        # State variables
        self.current_city = None
        self.visited_count = 0 
        self.current_observation = np.zeros(self.n_cities, dtype=int)
        self.action_mask = np.zeros(self.n_cities, dtype=int)
        self.return_buffer = np.zeros(self.n_cities + 1, dtype=int)
        self.current_precedence_count = None
        
        # Tracking
        self.reward_history = []
        self.best_route = []            
        self.best_cost = float('inf')
        
        # Q-Table
        self.Q = np.zeros((self.n_cities, self.n_cities))
        self.best_Q = None
        
    def reset(self):
            """Reset the episode state and return the initial observation."""
            self.current_city = -1 
            self.visited_count = 0
            self.current_precedence_count = self.base_precedence_count.copy()
            self.current_observation.fill(0) 
            self.action_mask.fill(0)     
            
            # Identify initially free cities (count == 0)
            free_cities = (self.current_precedence_count == 0)
            self.current_observation[free_cities] = 2 
            self.action_mask[free_cities] = 1     
            
            # Prepare return buffer
            self.return_buffer[:self.n_cities] = self.current_observation
            self.return_buffer[self.n_cities] = self.current_city
            return self.return_buffer.copy()

    def step(self, action=None):
            """Apply one transition and return the next SARSA interaction values."""
            prev_state = self.current_city 
            
            # Special Case: First action or no available actions
            if action is None:
                if not self.action_mask.any():
                    self.return_buffer[:self.n_cities] = self.current_observation
                    self.return_buffer[self.n_cities] = self.current_city
                    return self.return_buffer.copy(), 0.0, True, None
                
                # Initial Epsilon-greedy
                if np.random.rand() < self.epsilon:
                    valid_actions = np.flatnonzero(self.action_mask)
                    action = np.random.choice(valid_actions)
                else:
                    if prev_state == -1:
                        valid_actions = np.flatnonzero(self.action_mask)
                        action = np.random.choice(valid_actions)
                    else:
                        valid_actions = np.flatnonzero(self.action_mask)
                        q_values_valid = self.Q[prev_state, valid_actions]
                        action = valid_actions[np.argmax(q_values_valid)]
            
            next_city = action
            self.current_city = next_city
            self.visited_count += 1
            
            # Update State
            self.current_observation[next_city] = 1 # Mark as visited
            self.action_mask[next_city] = 0     # Remove from valid actions
            
            # Unlock dependencies
            dependents = self.unlocks[next_city]
            for dep in dependents:
                self.current_precedence_count[dep] -= 1
                if self.current_precedence_count[dep] == 0:
                    self.current_observation[dep] = 2 # Mark as available
                    self.action_mask[dep] = 1     
            
            # Calculate Reward (Negative Cost)
            cost = 0.0
            if prev_state != -1:
                cost = self.distance_matrix[prev_state, next_city]
            reward = -cost
            
            # Choose A' (Next Action) for SARSA logic
            next_action_chosen = None
            done = (self.visited_count == self.n_cities)
            
            if not done:
                if self.action_mask.any():
                    if np.random.rand() < self.epsilon:
                        valid_actions = np.flatnonzero(self.action_mask)
                        next_action_chosen = np.random.choice(valid_actions)
                    else:
                        valid_actions = np.flatnonzero(self.action_mask)
                        q_values_valid = self.Q[self.current_city, valid_actions]
                        next_action_chosen = valid_actions[np.argmax(q_values_valid)]
            
            # Q-Table Update (SARSA)
            if prev_state != -1:
                q_next_sa = 0.0
                if next_action_chosen is not None:
                    q_next_sa = self.Q[self.current_city, next_action_chosen]
                
                target = reward + self.gamma * q_next_sa
                old_q = self.Q[prev_state, next_city]
                self.Q[prev_state, next_city] = old_q + self.alpha * (target - old_q)
            
            self.return_buffer[:self.n_cities] = self.current_observation
            self.return_buffer[self.n_cities] = self.current_city
            return self.return_buffer.copy(), reward, done, next_action_chosen

    def render(self, window=100):
        """Save training and route diagnostics as a PDF figure."""
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'reward_history') or len(self.reward_history) == 0:
            print("No training data available.")
            return
        
        distances = np.array(self.reward_history)
        episodes = np.arange(len(distances))
        n = len(distances)
    
        if window > 1:
            dist_smooth = np.zeros(n)
            for i in range(n):
                start = max(0, i - window + 1)
                dist_smooth[i] = np.mean(distances[start:i+1])
        else:
            dist_smooth = distances
    
        plt.figure(figsize=(12, 6))
    
        plt.plot(episodes, dist_smooth, label='Distance (Moving Avg)', linewidth=2)
        plt.plot(episodes, distances, color='lightgray', alpha=0.5, label='Distance (Raw)')
    
        if self.target_solution > 0:
            plt.axhline(
                y=self.target_solution,
                color='r',
                linestyle='--',
                label=f'Target: {self.target_solution}'
            )
    
        plt.xlabel('Episodes', fontsize=16)
        plt.ylabel('Total Cost', fontsize=16)
        plt.title(f'Final RANDOM_SEARCH Masked AutoRL-SOP Training - {self.problem_name}', fontsize=18)
        plt.legend(fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.grid(True, alpha=0.3)
    
        plt.xlim(0, n-1)
        plt.margins(x=0)
    
        os.makedirs(TRAIN_PLOTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(TRAIN_PLOTS_PATH, f"{self.problem_name}.pdf"),
                    format='pdf', bbox_inches='tight')
        plt.show()
    
        print(f"Best route found: {' -> '.join(str(c) for c in self.best_route)}")
        print(f"Best distance obtained: {self.best_cost}")
    
        if self.target_solution > 0:
            diff = ((self.best_cost - self.target_solution) / self.target_solution) * 100
            print(f"Best known optimal: {self.target_solution}")
            print(f"Percentage difference (Gap): {diff:.2f}%")

def stop_study_callback(study, trial, target_val):
    """Stop the study after reaching the target solution."""
    if trial.value <= target_val:
        print(f"\n[CALLBACK] Target reached at trial {trial.number}! Cost: {trial.value}. Stopping Optimization.")
        study.stop()

def objective(trial, matrix_dist, precedences, target_val):
    """Evaluate one HPO configuration and return its best tour cost."""
    alpha = trial.suggest_float('alpha', 0.00, 1.00, step=0.01)
    gamma = trial.suggest_float('gamma', 0.00, 1.00, step=0.01)
    epsilon = trial.suggest_float('epsilon', 0.00, 1.00, step=0.01)
    
    # Fixed seed for trial reproducibility
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    
    env = SOPEnv(distance_matrix=matrix_dist,
                 precedences=precedences,
                 epsilon=epsilon,
                 alpha=alpha,
                 gamma=gamma,
                 n_training_eps=N_TRAIN_EPISODES,
                 target_solution=target_val)
    
    best_cost_found_in_trial = float('inf')

    # Rapid Training Loop
    for _ in range(env.n_training):
        env.reset()
        done = False
        current_action = None
        current_episode_cost = 0.0
        
        while not done:
            _, reward, done, next_action = env.step(action=current_action)
            current_episode_cost += -reward # Reward is negative cost
            current_action = next_action
        
        # Check if this route (via exploration or Q-Table) was the best in trial
        if current_episode_cost < best_cost_found_in_trial:
            best_cost_found_in_trial = current_episode_cost

    del env
    
    return best_cost_found_in_trial

if __name__ == "__main__":
    if not os.path.exists(BASE_PATH):
        print(f"CRITICAL ERROR: Path not found: {BASE_PATH}")
        print("Please check if the path is correct for your OS.")
        sys.exit()

    while True:
        print("\n" + "="*50)
        print("INSTANCE SELECTION (28 FROM PAPER)")
        print("="*50)
        for i, name in enumerate(SOP_INSTANCES):
            print(f"[{i+1:02d}] {name:<15}", end="")
            if (i + 1) % 4 == 0: print() 
        print()
            
        try:
            user_input = input("\nEnter instance number (1-28): ")
            choice = int(user_input)
            if 1 <= choice <= 28:
                file_name = SOP_INSTANCES[choice - 1]
                sop_file_path = os.path.join(BASE_PATH, file_name)
                key_name = file_name.replace('.sop', '')
                TARGET_SOLUTION = BEST_KNOWN_SOLUTIONS.get(key_name, 0)
                print(f"STOPPING TARGET: {TARGET_SOLUTION}")
                break
            else:
                print(">> Error: Invalid number.")
        except ValueError:
            print(">> Error: Please enter an integer.")

    print(f"\nSelected file: {sop_file_path}")

    try:
        problem = tsplib95.load(sop_file_path)
        # TSPlib95 loads edge weights with index offset 1 usually
        matrix_data = problem.as_dict()['edge_weights'][1:]
        GLOBAL_MATRIX_DIST = np.array(matrix_data) 
        N_CITIES = GLOBAL_MATRIX_DIST.shape[0]
        GLOBAL_PRECEDENCES = []
        
        # Precedence parsing: -1 implies a constraint
        for i in range(N_CITIES):
            for j in range(N_CITIES):
                if GLOBAL_MATRIX_DIST[i, j] == -1 and i != j:
                    GLOBAL_PRECEDENCES.append((j, i))
        print(f"Data loaded. Cities: {N_CITIES}, Precedences: {len(GLOBAL_PRECEDENCES)} (including all occurrences of '-1')")
    except Exception as e:
        print(f"ERROR LOADING FILE: {e}")
        sys.exit()

    # 2. Optuna Configuration and Execution
    start_total_time = time.time()
            
    print(f"\nStarting Random Search Optimization (Single Seed: {SEED_VALUE})...")
    print(f"Stopping conditions: Target {TARGET_SOLUTION} OR Timeout {OPTUNA_TIMEOUT}s")
            
    os.makedirs(RESULTS_PATH, exist_ok=True)
    db_name = os.path.join(RESULTS_PATH, f"optuna_results_{key_name}.db")
    db_url = f"sqlite:///{db_name}"
    sampler = optuna.samplers.RandomSampler(seed=SEED_VALUE)
    study = optuna.create_study(direction='minimize', sampler=sampler, storage=db_url, study_name=f"study_{key_name}", load_if_exists=True)
    
    # Wrappers to pass extra arguments
    objective_wrapper = lambda trial: objective(trial, GLOBAL_MATRIX_DIST, GLOBAL_PRECEDENCES, TARGET_SOLUTION)
    callback_wrapper = lambda study, trial: stop_study_callback(study, trial, TARGET_SOLUTION)
            
    try:
        study.optimize(objective_wrapper, 
                    n_trials=500, 
                    timeout=OPTUNA_TIMEOUT, 
                    n_jobs=1, 
                    callbacks=[callback_wrapper],
                    gc_after_trial=True)
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
            
    end_opt_time = time.time()
    opt_duration = end_opt_time - start_total_time
    
    print("\n" + "="*40)
    print("OPTIMIZATION RESULT")
    print("="*40)
    print(f"Best Cost: {study.best_value}")
    print(f"Best Trial (ID): {study.best_trial.number}")
    print(f"Last Executed Trial (ID): {study.trials[-1].number}")
    print(f"Best Params: {study.best_params}")
    
    # 3. Final Training
    print("\n" + "="*40)
    print("GENERATING FINAL MODEL WITH BEST PARAMS...")
    print("="*40)
    
    start_train_time = time.time()
    
    best_params = study.best_params
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    
    env = SOPEnv(distance_matrix=GLOBAL_MATRIX_DIST,
                precedences=GLOBAL_PRECEDENCES,
                epsilon=best_params['epsilon'],
                alpha=best_params['alpha'],
                gamma=best_params['gamma'],
                n_training_eps=N_TRAIN_EPISODES,
                problem_name=file_name,
                target_solution=TARGET_SOLUTION)

    target_reached = False 

    for ep in tqdm(range(env.n_training), desc="Final Training"):
        state = env.reset() 
        done = False
        episode_route = []
        total_reward = 0
        current_action = None

        while not done:
            state, reward, done, next_action = env.step(action=current_action) 
            total_reward += reward
            episode_route.append(env.current_city)
            current_action = next_action

        total_episode_cost = -total_reward
        env.reward_history.append(total_episode_cost)
            
        if total_episode_cost <= env.best_cost:
            env.best_cost = total_episode_cost
            env.best_route = episode_route[:]
            env.best_Q = env.Q.copy()
                    
            if env.best_cost <= TARGET_SOLUTION and TARGET_SOLUTION > 0 and not target_reached:
                print(f"\nTarget {TARGET_SOLUTION} reached (for the 1st time) at episode {ep} of final training!")
                target_reached = True

    end_train_time = time.time()
    train_duration = end_train_time - start_train_time
    total_duration = end_train_time - start_total_time

    env.render(window=100)

    print("\n" + "="*40)
    print("TIME REPORT")
    print("="*40)
    fmt = lambda t: f"{int(t // 60)} min {int(t % 60)} s {int((t % 1) * 1000):03d} ms"
    print(f"TOTAL Time:        {fmt(total_duration)}")
    print(f"Optimization Time: {fmt(opt_duration)}")
    print(f"Final Training Time: {fmt(train_duration)}")
