import minedojo
import minerl
import pickle
import pandas as pd
import numpy as np
import tarfile
import os

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

import sys

# Get the absolute path of VideoPreTraining and add it to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VideoPreTraining')))

from VideoPreTraining import lib
from VideoPreTraining.agent import MineRLAgent, ENV_KWARGS
import requests

STEP_BEFORE_SAVE = 1024
MAX_STEP = 10 # 300_000

def map_range(value):
    # Define your input range
    old_min = -10
    old_max = 10

    # Define your output range
    new_min = 0
    new_max = 24
    
    return round(((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min)

def convert_action(noop, action):
    new_action = noop.copy()

    new_action[0] = (action['forward'] + 2*action['back']) % 3
    new_action[1] = (action['left'] + 2*action['right']) % 3

    if((action['jump'] + action['sneak'] + action['sprint']) > 1 or (action['jump'] + action['sneak'] + action['sprint']) == 0):
        new_action[2] = 0
    elif(action['jump']):
        new_action[2] = 1
    elif(action['sneak']):
        new_action[2] = 2
    else:
        new_action[2] = 3

    new_action[3] = map_range(action['camera'][0][0])
    new_action[4] = map_range(action['camera'][0][1])

    return new_action

def main():
    model = "2x.model"
    weights = "rl-from-foundation-2x.weights"
    mock_env = HumanSurvival(**ENV_KWARGS).make()

    model_url = "https://openaipublic.blob.core.windows.net/minecraft-rl/models/" + model
    weights_url = "https://openaipublic.blob.core.windows.net/minecraft-rl/models/" + weights

    if not os.path.exists(model):
        response = requests.get(model_url)

        # Check if the request was successful
        if response.status_code == 200:
            with open(model, 'wb') as file:
                # pickle.dump(response.content,file)
                file.write(response.content)
            # print("Download complete!")
        else:
            print(f"Failed to download file, status code: {response.status_code}")
            return

    if not os.path.exists(weights):
        response = requests.get(weights_url)

        # Check if the request was successful
        if response.status_code == 200:
            with open(weights, 'wb') as file:
                # pickle.dump(response.content,file)
                file.write(response.content)
            # print("Download complete!")
        else:
            print(f"Failed to download file, status code: {response.status_code}")
            return

    # x print("---Launching MineRL enviroment (be patient)---")
    # mock_env.reset()

    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(mock_env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)
    
    mock_env.close()

    print("---Launching Minedojo enviroment (be patient)---")
    env = minedojo.make(task_id='survival', image_size=(360, 640))
    obs = env.reset()

    print("---Collecting data---")
    df = pd.DataFrame(columns=['pov', 'dx', 'dy', 'dz', 'dyaw', 'dpitch'])
    done = False
    id = 1
    noop = env.action_space.no_op()
    obs['pov'] = np.transpose(obs['rgb'], (1, 2, 0))
    prev_step = [obs['location_stats']['pos'][0], obs['location_stats']['pos'][1], obs['location_stats']['pos'][2], obs['location_stats']['yaw'][0], obs['location_stats']['pitch'][0]]
    while not done:
        action = agent.get_action(obs)
        action = convert_action(noop, action)
        obs, _, done, _ = env.step(action)
        obs['pov'] = np.transpose(obs['rgb'], (1, 2, 0))
        this_step = [obs['location_stats']['pos'][0], obs['location_stats']['pos'][1], obs['location_stats']['pos'][2], obs['location_stats']['yaw'][0], obs['location_stats']['pitch'][0]]
        delta = np.array(this_step) - np.array(prev_step)
        df.loc[len(df.index)] = [obs['pov'], delta[0], delta[1], delta[2], delta[3], delta[4]]
        prev_step = this_step
        if(id % STEP_BEFORE_SAVE == 0):
            print(f'---Saving {id-(STEP_BEFORE_SAVE-1)}-{id}---')
            df.to_pickle(f'data/data_{id-(STEP_BEFORE_SAVE-1)}-{id}.pkl.gz')

            df = pd.DataFrame(columns=['pov', 'dx', 'dy', 'dz', 'dyaw', 'dpitch'])
            print(f'---Saved---')
            # done = True

        id += 1

        if id > MAX_STEP:
            done = True

    env.close()

    print("---Done---")

if __name__ == '__main__':
    main()