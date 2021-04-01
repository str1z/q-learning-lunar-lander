import gym
from agent import Agent
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    n_episodes = 50000
    agent = Agent(lr=1e-3,
                  discount=0.99,
                  n_actions=env.action_space.n,
                  batch_size=64,
                  state_dims=env.observation_space.shape,
                  eps_dec=1e-4,
                  eps_min=0.01,
                  mem_size=1e6)

    for i in range(n_episodes):
        done = False
        state = env.reset()
        score = 0
        frame = 0

        while not done:
            action = agent.get_action(state)
            new_state, reward, done, info = env.step(action)

            agent.remember(action, reward, state, done)
            state = new_state

            if i > 10:
                env.render()

            agent.learn()
            agent.update_eps()

            score += reward
            frame += 1

        print(
            f"episode : {i} | memory : {agent.memory.counter / agent.memory.size} | score : {score} | epsilon : {agent.eps}")
