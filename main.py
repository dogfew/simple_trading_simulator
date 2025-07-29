import numpy as np
from matplotlib import pyplot as plt

from src.agent.simple_agent import SimpleAgent
from src.policies.simple_policy import SimpleGridPolicy
from src.data_processing.random_data import random_cycle_data, random_walk
from src.enviroment.simple_environment import SimpleEnvironment


def plot_data(data):
    time = range(len(data["prices"]))
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 6))
    axes[0].plot(time, data["prices"])
    axes[0].set_ylabel("Prices")
    axes[0].grid(True)

    axes[1].plot(time, data["welfare"])
    axes[1].set_ylabel("Welfare")
    axes[1].grid(True)

    axes[2].plot(time, data["position"])
    axes[2].set_ylabel("Position")
    axes[2].set_xlabel("Time")
    axes[2].grid(True)

    axes[3].plot(time, data["cash"])
    axes[3].set_ylabel("Cash")
    axes[3].set_xlabel("Time")
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # config
    batch_size = 1
    n_symbols = 2
    ticks = 60
    q1 = 0.2
    q2 = 1 - q1

    data = random_cycle_data((batch_size, n_symbols, ticks))  # [0.5, 2.5]
    buy_cutoff, sell_cutoff = np.quantile(data, [q1, q2])
    agent = SimpleAgent(batch_size, n_symbols)
    agent.set_default_state(cash=1)

    env = SimpleEnvironment(agent=agent, data=data)
    env.reset()

    policy = SimpleGridPolicy(
        batch_size, n_symbols, step_proportion=0.5, buy_cutoff=buy_cutoff, sell_cutoff=sell_cutoff
    )

    data = {
        "prices": [],
        "welfare": [],
        "position": [],
        "cash": []
    }

    for _ in range(100):
        actions = policy(env.observe(env.current_tick))
        obs, reward, terminated = env.step(actions)
        obs_market, obs_agent = obs

        data['welfare'].append(
            reward[0].item()
        )
        data['prices'].append(
            obs_market[0][0].item()
        )
        data['position'].append(
            obs_agent.quantities[0][0].item()
        )
        data['cash'].append(
            obs_agent.cash[0].item()
        )

        if np.any(terminated):
            break

    plot_data(data)


if __name__ == "__main__":
    main()
