import gym
from dqn import Agent
import numpy as np

def main():
    #make env and agent
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
                    eps_end=0.01, input_dims=[8], lr=0.0001)

    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            #ingame
            #get action from current view of game (observation)
            action = agent.choose_action(observation)
            #next frame
            observation_, reward, done, info = env.step(action)

            score += reward
            #store memory
            agent.store_transisation(observation, action, reward, observation_, done)
            agent.learn()

            #set next stage to current stage
            observation = observation_
        #append score and eps
        scores.append(score)
        eps_history.append(agent.epsilon)
    
        #print some nice statements
        avg_score = np.mean(scores[-100:])
        print(f'Episode: {i}   Score: {score}   Average Score: {avg_score}   Epsilon: {agent.epsilon}')

if __name__ == "__main__":
    main()