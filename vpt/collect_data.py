import minedojo
import minerl
import pickle
import pandas as pd
import numpy as np

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import MineRLAgent, ENV_KWARGS

env = minedojo.make(task_id='survival', image_size=(360, 640))

env.reset()

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

    print("---Launching MineRL enviroment (be patient)---")
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
    df = pd.DataFrame(columns=['pov', 'pos', 'yaw', 'pitch', 'reward', 'done', 'info'])
    done = False
    id = 1
    noop = env.action_space.no_op()
    obs['pov'] = np.transpose(obs['rgb'], (1, 2, 0))

    while not done:
        action = agent.get_action(obs)
        action = convert_action(noop, action)
        obs, reward, done, info = env.step(action)
        obs['pov'] = np.transpose(obs['rgb'], (1, 2, 0))

        df.loc[len(df.index)] = [obs['pov'], obs['location_stats']['pos'], obs['location_stats']['yaw'], obs['location_stats']['pitch'], reward, done, info]

        if(id % 10_000 == 0):
            print(f'---Saving {id-9_999}-{id}---')
            # df['pov'] = df['pov'].apply(lambda x: x.tobytes())
            df.to_pickle(f'data/data_{id-9_999}-{id}.pkl.gz')
            df = pd.DataFrame(columns=['pov', 'pos', 'yaw', 'pitch', 'reward', 'done', 'info'])
            print(f'---Saved---')
            done = True

        id += 1

if __name__ == '__main__':
    main()