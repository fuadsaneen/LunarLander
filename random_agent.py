# Import libraries.
import numpy as np
# import matplotlib.pyplot as plt

# Import simulations.
from simulations.lander import *

# Main function.
if __name__ == "__main__":

    print("\nPerforming random actions on the agent.\n")

    # Initialize environment.
    env = LunarLander()
    # Declare number of episodes.
    episodes = 100
    # Store score of each episode.
    scores = [0 for i in range(episodes)]

    # Do for each episode.
    for episode in range(1, episodes+1):

        # Reset variables.
        state = env.reset()
        done = False
        score = 0

        # Do till done.
        while not done:

            # Render.
            env.render()
            # Perform action from samples.
            action = env.action_space.sample()
            # Compute variables values.
            n_state, reward, done, info = env.step(action)
            # Update score.
            score += reward  
            # Store score in list.
            scores.append(score)  

        # Statistics of episode.    
        print('Episode:{} \nCurrent Score = {}'.format(episode, score))

        # Average score of previous episodes.
        is_solved = np.mean(scores[-episode:])

        # If solved, then exit.
        if is_solved > 200:
            print('Task Status: [Successfully Completed] \n')
            break
        # Else, print average score of all previous episodes.
        else:
            print("Average score = {0:.2f} \nTask Status: [Failed]\n".format(is_solved))

    # Plot score graph.
    plt.plot([i+1 for i in range(0, len(scores), 2)], scores[::2])
    plt.show()    

    # Close environment.
    env.close()