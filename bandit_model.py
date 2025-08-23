import numpy as np
from tqdm import trange
import random
import matplotlib.pyplot as plt 

class Bandit:
    """
    Implements a k-armed bandit problem with epsilon-greedy and UCB strategies.
    """
    def __init__(self, k_arm=10, epsilon=0.1, initial=0., step_size=0.1, sample_averages=False, UCB_param=None, true_reward=0., a=0.5):
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
        self.q_true = np.random.randn(self.k) + self.true_reward #[0, 1, 2]
        self.q_estimation = np.zeros(self.k) + self.initial #[0, 1, 2]
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)
        self.time = 0
        self.average_reward = 0

    def act(self):
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
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('action_reward_distribution.png')
    plt.close()


def explore_vs_exploit(runs=2000, time=1000):
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
    Showcase the impact of different values of c on UCB.
    """
    bandits = []
    c_values = [1, 2, 5]  # Different values of c for UCB
    for c in c_values:
        bandits.append(Bandit(epsilon=0, UCB_param=c, sample_averages=True))
    _, average_rewards = simulate(runs, time, bandits)

    plt.figure()
    for c, rewards in zip(c_values, average_rewards):
        plt.plot(rewards, label=f'UCB $c = {c}$')
    plt.plot(average_rewards[1], label=r'epsilon greedy $\epsilon = 0.1$')  # Use raw string
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('UCB_c_impact.png')
    plt.close()

if __name__ == '__main__':
    action_reward_distribution()
    explore_vs_exploit()
    initial_value_check()
    UCB_bandit()