import numpy as np
from tqdm import trange
import random
import matplotlib.pyplot as plt 

class Bandit:
    """
    Implements a k-armed bandit problem with epsilon-greedy and UCB strategies.

    Parameters:
    ----------
    k_arm : int, optional - Number of arms (actions) in the bandit (default: 10).
    epsilon : float, optional - Probability of exploration in epsilon-greedy (default: 0.1).
    initial : float, optional - Initial value for action-value estimates (default: 0.0).
    step_size : float, optional - Step size for updating action-value estimates (default: 0.1).
    sample_averages : bool, optional - Use sample averages for updates (default: False).
    UCB_param : float or None, optional - Parameter for UCB strategy; None disables UCB (default: None).
    true_reward : float, optional - Mean reward for each arm (default: 0.0).
    a : float, optional - Generalized step size parameter (0.5 < a < 1, default: 0.5).
    """
    def __init__(self, 
                 k_arm=10, 
                 epsilon=0.1, 
                 initial=0., 
                 step_size=0.1, 
                 sample_averages=False, 
                 UCB_param=None, 
                 true_reward=0., 
                 a=0.5):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)  # 0, 1, 2
        self.time = 0
        self.UCB_param = UCB_param
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        self.a = a  # Parameter for generalized step size (0.5 < a < 1)

    def reset(self):
        """
        Resets the bandit to its initial state.

        Attributes:
        ----------
        q_true : numpy.ndarray - True rewards for each arm, sampled from a normal distribution.
        q_estimation : numpy.ndarray - Estimated rewards for each arm, initialized to `initial`.
        action_count : numpy.ndarray - Number of times each arm has been selected.
        best_action : int - Index of the arm with the highest true reward.
        time : int - Tracks the number of steps taken.
        average_reward : float - Tracks the average reward over time.
        """
        self.q_true = np.random.randn(self.k) + self.true_reward #[0, 1, 2]  # randn gives standrd normal distribution
        self.q_estimation = np.zeros(self.k) + self.initial #[0, 1, 2]
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time = 0
        self.average_reward = 0

    def act(self):
        """
        Selects an action based on the current strategy.

        Returns:
        -------
        int - Index of the selected action.
        """
        # UCB (Upper Confidence Bound) strategy
        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            best_action = np.random.choice(np.where(UCB_estimation == q_best)[0])
            return best_action
        
        # Epsilon-Greedy strategy (default)
        if np.random.rand() < self.epsilon:
            action_id = np.random.choice(self.indices)
            return action_id

        q_best = np.max(self.q_estimation)
        best_action = np.random.choice(np.where(self.q_estimation == q_best)[0])
        return best_action

    def step(self, action, external_reward=None):
        """
        Takes an action and updates the bandit's state.

        Parameters:
        ----------
        action : int - Index of the action to take.
        external_reward : float or None, optional - Reward provided externally; None samples from distribution.

        Returns:
        -------
        float - Reward received for the selected action.
        """
        if external_reward is not None:
            reward = external_reward
        else:
            reward = np.random.randn() + self.q_true[action]
            
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
            # Generalized decreasing step size: 1 / (n+1)^a
            step_size = 1 / (self.action_count[action] ** self.a)
            self.q_estimation[action] += step_size * (reward - self.q_estimation[action])
        else:
            # Constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        
        return reward
    
def simulate(runs, time, bandits):
    """
    Simulates multiple runs of the bandit problem.

    Parameters:
    ----------
    runs : int - Number of independent runs to simulate.
    time : int - Number of steps per run.
    bandits : list of Bandit - List of Bandit instances to simulate.

    Returns:
    -------
    tuple - (mean_best_action_counts, mean_rewards):
        mean_best_action_counts : numpy.ndarray - Mean percentage of optimal actions over time.
        mean_rewards : numpy.ndarray - Mean rewards received over time.
    """
    rewards = np.zeros((len(bandits), runs, time))  # 3, 2000, 1000
    best_action_counts = np.zeros(rewards.shape)
    print(best_action_counts.shape)
    # import sys; sys.exit()
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def action_reward_distribution():
    """
    Visualizes the reward distribution for each action using a violin plot.
    """
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('action_reward_distribution.png')
    plt.close()


def explore_vs_exploit(runs=2000, time=1000):
    """
    Compares the performance of different epsilon values in the epsilon-greedy strategy.

    Parameters:
    ----------
    runs : int, optional - Number of independent runs to simulate (default: 2000).
    time : int, optional - Number of steps per run (default: 1000).
    """
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label=r'$\epsilon = %.02f$' % (eps))  # Use raw string
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label=r'$\epsilon = %.02f$' % (eps))  # Use raw string
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('explore_vs_exploit.png')
    plt.close()


def initial_value_check(runs=2000, time=1000):
    """
    Compares the impact of optimistic initial values on exploration.

    Parameters:
    ----------
    runs : int, optional - Number of independent runs to simulate (default: 2000).
    time : int, optional - Number of steps per run (default: 1000).
    """
    bandits = []
    bandits.append(Bandit(epsilon=0, initial=5, step_size=0.1))
    bandits.append(Bandit(epsilon=0.1, initial=0, step_size=0.1))
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label=r'$\epsilon = 0, q = 5$')  # Use raw string
    plt.plot(best_action_counts[1], label=r'$\epsilon = 0.1, q = 0$')  # Use raw string
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('initial_value_check.png')
    plt.close()

def UCB_bandit(runs=2000, time=1000):
    """
    Demonstrates the impact of different values of c on the UCB strategy and compares it with epsilon-greedy.

    Parameters:
    ----------
    runs : int, optional - Number of independent runs to simulate (default: 2000).
    time : int, optional - Number of steps per run (default: 1000).
    """
    c_values = [2, 5]  # Different values of c for UCB
    epsilon = 0.1  # Epsilon value for epsilon-greedy
    bandits = [Bandit(epsilon=0, UCB_param=c, sample_averages=True) for c in c_values]
    bandits.append(Bandit(epsilon=epsilon, sample_averages=True))  # Add epsilon-greedy bandit
    labels = [f'UCB $c = {c}$' for c in c_values] + [f'Epsilon-Greedy $\\epsilon = {epsilon}$']

    _, average_rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 6))
    for label, rewards in zip(labels, average_rewards):
        plt.plot(rewards, label=label)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.title('Comparison of UCB and Epsilon-Greedy Strategies')
    plt.legend()

    plt.savefig('UCB_vs_epsilon_greedy.png')
    plt.close()

def study_a_impact(runs=2000, time=1000):
    """
    Studies the impact of changing the parameter 'a' (generalized step size) on performance.

    Parameters:
    ----------
    runs : int, optional - Number of independent runs to simulate (default: 2000).
    time : int, optional - Number of steps per run (default: 1000).
    """
    a_values = [0.5, 0.6, 0.7, 0.8, 0.9]  # Different values of 'a' to test
    bandits = [Bandit(epsilon=0.2, sample_averages=True, a=a) for a in a_values]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    # Plot average rewards for each value of 'a'
    plt.subplot(2, 1, 1)
    for a, reward in zip(a_values, rewards):
        plt.plot(reward, label=f'$a = {a}$')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    # Plot percentage of optimal actions for each value of 'a'
    plt.subplot(2, 1, 2)
    for a, counts in zip(a_values, best_action_counts):
        plt.plot(counts, label=f'$a = {a}$')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.savefig('a_impact.png')
    plt.close()

def compare_with_without_a(runs=2000, time=1000):
    """
    Compares the performance of the bandit algorithm with and without the parameter 'a'.

    Parameters:
    ----------
    runs : int, optional - Number of independent runs to simulate (default: 2000).
    time : int, optional - Number of steps per run (default: 1000).
    """
    a_values = [0.5, 0.7, 0.9]  # Different values of 'a' to test
    bandits_with_a = [Bandit(epsilon=0.1, sample_averages=True, a=a) for a in a_values]
    bandit_without_a = Bandit(epsilon=0.1, sample_averages=True, a=1.0)  # Standard sample average (a = 1)
    bandits = bandits_with_a + [bandit_without_a]
    labels = [f'With $a = {a}$' for a in a_values] + ['Without $a$ (Standard)']

    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    # Plot average rewards for each configuration
    plt.subplot(2, 1, 1)
    for label, reward in zip(labels, rewards):
        plt.plot(reward, label=label)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    # Plot percentage of optimal actions for each configuration
    plt.subplot(2, 1, 2)
    for label, counts in zip(labels, best_action_counts):
        plt.plot(counts, label=label)
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.savefig('compare_with_without_a.png')
    plt.close()

if __name__ == '__main__':
    # action_reward_distribution()
    # explore_vs_exploit()
    # initial_value_check()
    UCB_bandit()
    study_a_impact()
    compare_with_without_a()