# Import libraries.
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.agents import DQNAgent

# Build agent.
def build_dqn_agent(model,actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit = 50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions = actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn