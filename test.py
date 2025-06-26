from utils import *
from sac import *

if __name__ == "__main__":
    env_name = "HalfCheetah-v5"
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim, reward_scale=2, tau=0.005)
    agent.load_models()

    done = False
    obs, _ = env.reset()
    for i in range(4000):
        action = agent.choose_action(obs.tolist())
        obs, reward, terminated, truncated, _ = env.step(action)
    env.close()