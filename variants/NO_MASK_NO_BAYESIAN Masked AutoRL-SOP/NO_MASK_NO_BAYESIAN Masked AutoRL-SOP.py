"""AutoRL-SOP ablation using resampling and HyperTuningSK optimization."""

import tsplib95
import numpy as np
from tqdm import tqdm
import random
import time
import os
import sys
import csv
import warnings
from scipy import stats

# GENERAL CONFIGURATION
SEED_VALUE = 42
FINAL_EPISODES = 10_000        # episodes for the final training
HPO_EPISODES = 200             # episodes per evaluation during HPO
EPOCHS = 5                     # repetitions per candidate value
HP_VALUES = [0.05, 0.25, 0.45, 0.65, 0.85]   # candidate grid

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
    def __init__(self, distance_matrix, epsilon, alpha, gamma, n_training_eps, problem_name="SOP", target_solution=0):
        """Initialize the environment state and SARSA parameters."""
        self.distance_matrix = distance_matrix
        self.n_cities = distance_matrix.shape[0]
        self.target_solution = target_solution

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_training = n_training_eps
        self.problem_name = problem_name

        self.current_city = None
        self.visited_count = 0
        self.visited = np.zeros(self.n_cities, dtype=bool)

        self.reward_history = []
        self.best_route = []
        self.best_cost = float('inf')

        self.Q = np.zeros((self.n_cities, self.n_cities))
        self.best_Q = None

    def reset(self):
            """Reset the episode state and return the initial observation."""
            self.current_city = -1
            self.visited_count = 0
            self.visited.fill(False)

    def _select_action_resampling(self, from_city):
            """
            Select a feasible action by resampling. An epsilon-greedy candidate is
            drawn from the still-unvisited nodes; if its precedence constraints are
            not all satisfied it is dropped and a new candidate is drawn, until a
            feasible node is found (or None if none remains). Feasibility is tested
            by scanning the distance matrix for unmet precedence entries (-1).
            """
            available = list(np.flatnonzero(~self.visited))

            # epsilon-greedy decision drawn once per selection and reused while resampling
            greed_sel = np.random.rand()

            while available:
                if greed_sel < self.epsilon or from_city == -1:
                    a = int(np.random.choice(available))
                else:
                    q_values = self.Q[from_city, available]
                    a = int(available[int(np.argmax(q_values))])

                feasible = True
                for j in available:
                    if j != a and self.distance_matrix[a, j] == -1:
                        feasible = False

                if feasible:
                    return a  # accept

                available.remove(a)  # infeasible -> drop and resample

            return None

    def step(self, action=None):
            """Apply one transition and return the next SARSA interaction values."""
            prev_state = self.current_city

            # First action of the episode (or no feasible action left)
            if action is None:
                action = self._select_action_resampling(prev_state)
                if action is None:
                    return 0.0, True, None

            next_city = action
            self.current_city = next_city
            self.visited_count += 1
            self.visited[next_city] = True

            # Reward = negative travelled cost
            cost = 0.0
            if prev_state != -1:
                cost = self.distance_matrix[prev_state, next_city]
            reward = -cost

            # Choose A' (next action) for the SARSA update
            next_action_chosen = None
            done = (self.visited_count == self.n_cities)
            if not done:
                next_action_chosen = self._select_action_resampling(self.current_city)

            # Q-Table update (SARSA)
            if prev_state != -1:
                q_next_sa = 0.0
                if next_action_chosen is not None:
                    q_next_sa = self.Q[self.current_city, next_action_chosen]

                target = reward + self.gamma * q_next_sa
                old_q = self.Q[prev_state, next_city]
                self.Q[prev_state, next_city] = old_q + self.alpha * (target - old_q)

            return reward, done, next_action_chosen

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
            plt.axhline(y=self.target_solution, color='r', linestyle='--',
                        label=f'Target: {self.target_solution}')

        plt.xlabel('Episodes', fontsize=16)
        plt.ylabel('Total Cost', fontsize=16)
        plt.title(f'Final NO_MASK_NO_BAYESIAN Masked AutoRL-SOP Training - {self.problem_name}', fontsize=18)
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


# A single RL training run. Each call builds a fresh environment (Q-table reset
# to zero). The global RNG stream is intentionally NOT reset between calls, so the
# EPOCHS repetitions of a given candidate value are distinct random runs; this is
# what provides the within-group variance required by the Scott-Knott statistics.
def run_training(matrix_dist, epsilon, alpha, gamma, n_episodes,
                 problem_name="SOP", target_solution=0, track_history=False):
    """Train one SARSA configuration and return its observed performance."""
    env = SOPEnv(distance_matrix=matrix_dist,
                 epsilon=epsilon,
                 alpha=alpha,
                 gamma=gamma,
                 n_training_eps=n_episodes,
                 problem_name=problem_name,
                 target_solution=target_solution)

    best_cost = float('inf')
    best_episode = -1
    best_route = []

    iterator = range(n_episodes)
    if track_history:
        iterator = tqdm(iterator, desc="Final Training")

    for ep in iterator:
        env.reset()
        done = False
        current_action = None
        episode_cost = 0.0
        episode_route = []

        while not done:
            reward, done, next_action = env.step(action=current_action)
            episode_cost += -reward
            episode_route.append(env.current_city)
            current_action = next_action

        if track_history:
            env.reward_history.append(episode_cost)

        if episode_cost <= best_cost:
            best_cost = episode_cost
            best_episode = ep
            best_route = episode_route[:]

    if track_history:
        env.best_cost = best_cost
        env.best_route = best_route
        return best_cost, best_episode, env

    return best_cost, best_episode, None


# SCOTT-KNOTT CLUSTERING (Scott & Knott, 1974). Recursively partitions a set of
# treatment means into homogeneous, non-overlapping groups using the
# likelihood-ratio criterion lambda = (pi/(2*(pi-2))) * B0 / sigma2_0, compared
# against a chi-square critical value with df = m/(pi-2), where m is the number
# of means in the current node. The variance estimate sigma2_0 is recomputed at
# every node from the means it contains and the (fixed) residual term (the
# node-local Scott-Knott estimator). Returns an integer label per treatment.
def scott_knott(data_groups, alpha_level=0.05):
    """Cluster candidate treatments with the Scott-Knott procedure."""
    k = len(data_groups)
    means = np.array([np.mean(g) for g in data_groups], dtype=float)
    reps = np.array([len(g) for g in data_groups], dtype=float)

    n_total = int(np.sum(reps))
    ss_within = float(np.sum([np.sum((np.asarray(g) - np.mean(g)) ** 2) for g in data_groups]))
    df_error = n_total - k
    mse = ss_within / df_error if df_error > 0 else 0.0

    r_bar = np.mean(reps)
    mMSE = mse / r_bar          # residual term, fixed across nodes (MSE / replications)

    PI = np.pi
    const = PI / (2.0 * (PI - 2.0))

    labels = np.zeros(k, dtype=int)
    next_label = [0]
    order = np.argsort(means)

    def recurse(idx_sorted):
        """Recursively partition an ordered set of treatment means."""
        m = len(idx_sorted)
        cur_label = next_label[0]

        # node-local variance estimate (only the means in this node)
        node_means = means[idx_sorted]
        node_var = float(np.sum((node_means - np.mean(node_means)) ** 2))
        sigma2_0 = (node_var + df_error * mMSE) / (m + df_error)

        if m <= 1 or sigma2_0 <= 0:
            for i in idx_sorted:
                labels[i] = cur_label
            next_label[0] += 1
            return

        total = np.sum(node_means)
        best_B0 = -1.0
        best_split = -1
        for s in range(1, m):
            t1 = np.sum(node_means[:s])
            t2 = total - t1
            B = (t1 * t1) / s + (t2 * t2) / (m - s) - (total * total) / m
            if B > best_B0:
                best_B0 = B
                best_split = s

        lam = const * best_B0 / sigma2_0
        df_chi = m / (PI - 2.0)
        crit = stats.chi2.ppf(1.0 - alpha_level, df_chi)

        if lam > crit:
            recurse(idx_sorted[:best_split])
            recurse(idx_sorted[best_split:])
        else:
            for i in idx_sorted:
                labels[i] = cur_label
            next_label[0] += 1

    recurse(order)
    return labels


# HyperTuningSK (ANOVA + KS normality + Bartlett homogeneity + Scott-Knott).
# Returns the list of candidate VALUES recommended for the tuned hyperparameter
# (the best
# Scott-Knott group, i.e. lowest mean distance). If the model is not adequate
# (non-normal residuals) or there is no significant difference, a single random
# value in [min, max] is returned.
def hyper_tuning_sk(values, data_groups):
    """Select candidate values through the HyperTuningSK workflow."""
    group_means = np.array([np.mean(g) for g in data_groups])
    residuals = np.concatenate([np.asarray(g, dtype=float) - np.mean(g) for g in data_groups])

    # The statistical tests can warn on degenerate input (e.g. a trivial
    # instance where every repetition reaches the same optimum, yielding a
    # constant group). Such cases are handled by falling back to a random
    # recommendation, so the warnings are suppressed here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            try:
                paov = stats.f_oneway(*data_groups).pvalue
            except Exception:
                paov = 1.0
            if not np.isfinite(paov):
                paov = 1.0

            sd = np.std(residuals, ddof=1)
            if sd > 0:
                pks = stats.kstest(residuals, 'norm', args=(np.mean(residuals), sd)).pvalue
            else:
                pks = 0.0

            try:
                pbt = stats.bartlett(*data_groups).pvalue
            except Exception:
                pbt = float('nan')

    Ha = 0
    if pks < 0.05:
        print("Non-normal residuals")
    else:
        print("Normal residuals")
        if paov < 0.05:
            print("Alternative hypothesis (Ha): hyperparameters differ significantly")
            Ha = 1
        else:
            print("Null hypothesis (Ho): no significant difference between hyperparameters")

    print(f"Normality (pks): {pks:.4g} | Homogeneity (pbt): {pbt:.4g} | ANOVA p-value: {paov:.4g}")

    if Ha == 1:
        labels = scott_knott(data_groups, alpha_level=0.05)
        unique_labels = np.unique(labels)
        group_avg = {lab: np.mean(group_means[labels == lab]) for lab in unique_labels}
        best_label = min(group_avg, key=group_avg.get)
        idx = np.flatnonzero(labels == best_label)
        recommended = [values[i] for i in idx]
        print(f"Scott-Knott -> recommended parameters: {recommended}")
    else:
        recommended = [np.random.uniform(min(values), max(values))]
        print(f"No difference -> random recommended parameter: {recommended}")

    return recommended


# Lightweight tracker for reporting the best HPO evaluation (no stopping).
class Tracker:
    """Track the best configuration observed during sequential tuning."""
    def __init__(self):
        """Initialize an empty best-observation record."""
        self.evaluations = 0
        self.best_cost = float('inf')
        self.best_params = None

    def record(self, cost, params):
        """Record a configuration when it improves the best observed cost."""
        self.evaluations += 1
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params


# Columns of the complete optimizer-log CSV (analogous to an Optuna trials table:
# one row per RL evaluation, plus per-phase recommendation/setup rows).
LOG_COLUMNS = ["stage", "eval_index", "tuned_param", "candidate_value", "epoch",
               "epsilon", "alpha", "gamma", "n_episodes", "min_distance",
               "best_episode", "note"]


def log_row(**kw):
    """Create a normalized row for the optimizer CSV log."""
    row = {c: "" for c in LOG_COLUMNS}
    row.update(kw)
    return row


def run_phase(phase_name, phase_tag, tuned_values, fixed_eps, fixed_alpha, fixed_gamma,
              tuned_kind, matrix_dist, target_solution, tracker, log):
    """Evaluate one sequential hyperparameter-tuning phase."""
    print("\n" + "-" * 60)
    print(phase_name)
    print("-" * 60)

    data_groups = []
    for val in tuned_values:
        epoch_costs = []
        for epoch_i in range(EPOCHS):
            eps = val if tuned_kind == 'epsilon' else fixed_eps
            alp = val if tuned_kind == 'alpha' else fixed_alpha
            gam = val if tuned_kind == 'gamma' else fixed_gamma

            cost, best_ep, _ = run_training(matrix_dist, eps, alp, gam,
                                            HPO_EPISODES, target_solution=target_solution)
            epoch_costs.append(cost)
            tracker.record(cost, (eps, alp, gam))
            log.append(log_row(stage=phase_tag, eval_index=tracker.evaluations,
                               tuned_param=tuned_kind, candidate_value=f"{val:.2f}",
                               epoch=epoch_i + 1, epsilon=f"{eps:.2f}", alpha=f"{alp:.2f}",
                               gamma=f"{gam:.2f}", n_episodes=HPO_EPISODES,
                               min_distance=cost, best_episode=best_ep))
            print(f"  {tuned_kind}={val:.2f} | eps={eps:.2f} alpha={alp:.2f} gamma={gam:.2f} -> min dist={cost:.2f}")
        data_groups.append(np.array(epoch_costs, dtype=float))

    recommended = hyper_tuning_sk(tuned_values, data_groups)
    log.append(log_row(stage="recommendation", tuned_param=tuned_kind,
                       note="recommended " + tuned_kind + " set = "
                            + repr([round(float(x), 4) for x in recommended])))
    return recommended


if __name__ == "__main__":
    if not os.path.exists(BASE_PATH):
        print(f"CRITICAL ERROR: Path not found: {BASE_PATH}")
        print("Please check if the path is correct for your OS.")
        sys.exit()

    os.makedirs(RESULTS_PATH, exist_ok=True)

    while True:
        print("\n" + "=" * 50)
        print("INSTANCE SELECTION (28 FROM PAPER)")
        print("=" * 50)
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
        matrix_data = problem.as_dict()['edge_weights'][1:]
        GLOBAL_MATRIX_DIST = np.array(matrix_data)
        N_CITIES = GLOBAL_MATRIX_DIST.shape[0]
        n_precedences = int(np.count_nonzero(GLOBAL_MATRIX_DIST == -1))
        print(f"Data loaded. Cities: {N_CITIES}, Precedences: {n_precedences} (occurrences of '-1')")
    except Exception as e:
        print(f"ERROR LOADING FILE: {e}")
        sys.exit()

    # OPTIMIZATION STAGE: HyperTuningSK + Scott-Knott (3 sequential phases, full)
    start_total_time = time.time()

    print(f"\nStarting HyperTuningSK Optimization (Single Seed: {SEED_VALUE})...")

    # Single seed for the whole run: the optimizer meta-decisions (the per-phase
    # fixed hyperparameters and the random fallback) and the RL runs all draw from
    # the same NumPy stream, set once here and not reset between runs. This keeps
    # the optimization deterministic across executions while the EPOCHS repetitions
    # of a candidate remain distinct draws (the variance Scott-Knott needs).
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    tracker = Tracker()
    optimizer_log = []

    alpha1 = np.random.uniform(min(HP_VALUES), max(HP_VALUES))
    gamma1 = np.random.uniform(min(HP_VALUES), max(HP_VALUES))
    optimizer_log.append(log_row(stage="phase_setup", tuned_param="epsilon",
                                 note=f"Phase 1 fixed alpha={alpha1:.4f}, gamma={gamma1:.4f}"))
    esk = run_phase("Phase 1: tuning EPSILON (e-greedy)", "HPO_epsilon", HP_VALUES,
                    None, alpha1, gamma1, 'epsilon',
                    GLOBAL_MATRIX_DIST, TARGET_SOLUTION, tracker, optimizer_log)

    gamma2 = np.random.uniform(min(HP_VALUES), max(HP_VALUES))
    e2 = np.random.uniform(min(esk), max(esk))
    optimizer_log.append(log_row(stage="phase_setup", tuned_param="alpha",
                                 note=f"Phase 2 fixed epsilon={e2:.4f}, gamma={gamma2:.4f}"))
    ask = run_phase("Phase 2: tuning ALPHA (learning rate)", "HPO_alpha", HP_VALUES,
                    e2, None, gamma2, 'alpha',
                    GLOBAL_MATRIX_DIST, TARGET_SOLUTION, tracker, optimizer_log)

    alpha3 = np.random.uniform(min(ask), max(ask))
    e3 = np.random.uniform(min(esk), max(esk))
    optimizer_log.append(log_row(stage="phase_setup", tuned_param="gamma",
                                 note=f"Phase 3 fixed epsilon={e3:.4f}, alpha={alpha3:.4f}"))
    gsk = run_phase("Phase 3: tuning GAMMA (discount factor)", "HPO_gamma", HP_VALUES,
                    e3, alpha3, None, 'gamma',
                    GLOBAL_MATRIX_DIST, TARGET_SOLUTION, tracker, optimizer_log)

    end_opt_time = time.time()
    opt_duration = end_opt_time - start_total_time

    print("\n" + "=" * 40)
    print("OPTIMIZATION RESULT (HyperTuningSK)")
    print("=" * 40)
    print(f"Recommended EPSILON set: {esk}")
    print(f"Recommended ALPHA   set: {ask}")
    print(f"Recommended GAMMA   set: {gsk}")
    print(f"HP evaluations performed: {tracker.evaluations}")
    print(f"Best cost during HPO: {tracker.best_cost} with params (eps,alpha,gamma)={tracker.best_params}")

    # FINAL STAGE: grid over recommended (esk x ask x gsk), FINAL_EPISODES each. The overall best is reported.
    print("\n" + "=" * 40)
    print("GENERATING FINAL MODEL WITH RECOMMENDED PARAMS...")
    print("=" * 40)

    start_train_time = time.time()

    final_combos = [(e_val, a_val, g_val)
                    for e_val in esk for a_val in ask for g_val in gsk]

    solution_table = []
    overall_best = {'cost': float('inf'), 'params': None, 'episode': -1, 'env': None}
    final_eval_index = tracker.evaluations

    for (e_val, a_val, g_val) in final_combos:
        # Final stage uses the fixed seed, i.e. the same 'world of randomness' as
        # the Masked model: every combination is trained from the identical RNG
        # state, so the final evaluation is deterministic per environment, fairly
        # comparable across combinations/instances/models, and reproducible if the
        # final stage is re-run on its own. (The optimization stage above keeps the
        # continuous RNG that the Scott-Knott variance analysis requires.)
        random.seed(SEED_VALUE)
        np.random.seed(SEED_VALUE)
        best_cost, best_ep, env = run_training(
            GLOBAL_MATRIX_DIST,
            e_val, a_val, g_val, FINAL_EPISODES,
            problem_name=file_name, target_solution=TARGET_SOLUTION,
            track_history=True)

        solution_table.append((e_val, a_val, g_val, best_cost, best_ep))
        final_eval_index += 1
        optimizer_log.append(log_row(stage="final_grid", eval_index=final_eval_index,
                                     epsilon=f"{e_val:.2f}", alpha=f"{a_val:.2f}",
                                     gamma=f"{g_val:.2f}", n_episodes=FINAL_EPISODES,
                                     min_distance=best_cost, best_episode=best_ep))
        print(f"  eps={e_val:.2f} alpha={a_val:.2f} gamma={g_val:.2f} -> "
              f"min dist={best_cost:.2f} (best episode {best_ep})")

        if best_cost < overall_best['cost']:
            overall_best.update({'cost': best_cost, 'params': (e_val, a_val, g_val),
                                 'episode': best_ep, 'env': env})

    end_train_time = time.time()
    train_duration = end_train_time - start_train_time
    total_duration = end_train_time - start_total_time

    if overall_best['env'] is not None:
        overall_best['env'].render(window=100)

    be, ba, bg = overall_best['params']

    # Overall best, recorded as the last row of the complete optimizer log.
    optimizer_log.append(log_row(stage="overall_best", epsilon=f"{be:.2f}",
                                 alpha=f"{ba:.2f}", gamma=f"{bg:.2f}",
                                 n_episodes=FINAL_EPISODES, min_distance=overall_best['cost'],
                                 best_episode=overall_best['episode'],
                                 note="best combination over the final grid"))

    # Lowest distance seen at ANY stage (HPO evaluations or final training). The
    # combination ranked best by HPO may not reproduce the lowest value in the
    # final stage, so the overall best ever observed is reported as well.
    best_observed = min(tracker.best_cost, overall_best['cost'])
    best_observed_stage = "HPO" if tracker.best_cost < overall_best['cost'] else "final training"
    optimizer_log.append(log_row(stage="best_observed", min_distance=best_observed,
                                 note=f"lowest distance seen at any stage (from {best_observed_stage})"))

    # Complete optimizer log: every HPO evaluation, per-phase setup/recommendation
    # rows, the full final grid and the overall best. This is the full record of
    # the run, so the optimization does not need to be repeated to inspect it.
    csv_path = os.path.join(RESULTS_PATH, f"hypertuningsk_results_{key_name}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for row in optimizer_log:
            writer.writerow(row)

    print("\n" + "=" * 40)
    print("FINAL RESULT")
    print("=" * 40)
    print(f"Best Cost (final training): {overall_best['cost']}")
    print(f"Best Params: {{'epsilon': {be}, 'alpha': {ba}, 'gamma': {bg}}}")
    print(f"Best episode (final training): {overall_best['episode']}")
    print(f"Max episodes (final training): {FINAL_EPISODES}")
    print(f"Final-grid combinations evaluated: {len(solution_table)}")
    print(f"Best value observed at any stage: {best_observed} (from {best_observed_stage})")
    if TARGET_SOLUTION > 0:
        gap = ((overall_best['cost'] - TARGET_SOLUTION) / TARGET_SOLUTION) * 100
        gap_observed = ((best_observed - TARGET_SOLUTION) / TARGET_SOLUTION) * 100
        print(f"Best known optimal: {TARGET_SOLUTION} | Gap (final): {gap:.2f}% | Gap (best observed): {gap_observed:.2f}%")
    print(f"Results table saved to: {csv_path}")

    print("\n" + "=" * 40)
    print("TIME REPORT")
    print("=" * 40)
    fmt = lambda t: f"{int(t // 60)} min {int(t % 60)} s {int((t % 1) * 1000):03d} ms"
    print(f"TOTAL Time:        {fmt(total_duration)}")
    print(f"Optimization Time: {fmt(opt_duration)}")
    print(f"Final Training Time: {fmt(train_duration)}")
