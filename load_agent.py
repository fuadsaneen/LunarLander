# Import libraries.
from tensorflow.keras.optimizers import Adam

# Import architectures.
from architectures.model_architecure import *
from architectures.dqn_architecture import *

# Import simulations.
from simulations.lander import *
from simulations.main_engine_failure import *
from simulations.side_engine_failure import *

# Main function.
if __name__ == "__main__":

    # Choose simulation mode.
    choice = int(input("\nChoose simulation mode:\n\n 1: Normal Mode.\n 2: Side Engine Failure.\n 3: Main Engine Failure.\n\nEnter your choice: "))
    if choice == 1:
        env = LunarLander()
    elif choice == 2:
        env = SideEngineFailure()
    elif choice == 3:
        env = MainEngineFailure()
    else:
        print("\nPlease enter a valid choice.\n")
        exit(0)

    # Initialize state and action space.
    states = env.observation_space.shape[0]
    actions = env.action_space.n

    # Initialize model.
    model = build_model(states, actions)

    # Initialize agent.
    dqn = build_dqn_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Load weights into DQN.
    steps = "agent"
    file_name = str(steps) + '.h5'
    dqn.load_weights(file_name)
    print("\nThe agent trained for " + str(282) + " times is loaded from disk.\n")

    # Test agent.
    print("\nEpisode Reports: \n")
    _ = dqn.test(env, nb_episodes=100, nb_max_episode_steps=3000, visualize=True)