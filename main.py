import gym
import numpy as np
from agent import Agent
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    n_episodes = 50000
    agent = Agent(lr=1e-3,
                  discount=0.9,
                  n_actions=env.action_space.n,
                  batch_size=64,
                  state_dims=env.observation_space.shape,
                  eps_dec=1e-5,
                  eps_min=0.01,
                  mem_size=1e6)

    for i in range(n_episodes):
        done = False
        data = env.reset()
        score = 0

        while not done:
            action = agent.get_action(data)
            state, reward, done, info = env.step(action)
            score += reward

            if agent.eps < 0.5:
                env.render()

            agent.learn()
            agent.update_eps()

            agent.remember(action, reward, state, done)

        print(
            f"episode : {i} | memory : {agent.memory.counter / agent.memory.size} | score : {score} | epsilon : {agent.eps}")
