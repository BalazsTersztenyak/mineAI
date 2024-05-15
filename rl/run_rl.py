import gym
from models.encoder import Autoencoder
from models.transformer import Transformer
import torch
import torch.nn as nn
# import numpy as np
from queue import Queue
from tqdm import tqdm
from mineclip import MineCLIP
# import hydra
# from omegaconf import OmegaConf
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.autograd import profiler
import pickle
from torch.utils.checkpoint import checkpoint

# Needed to register MineRLHumanSurvival-v0 environment to gym
import minerl.herobraine.envs

ENCODER_PATH = 'data/checkpoints/model_199.ckpt'
RL_PATH = None #'rl_model_delta.ckpt'
DELTA_PATH = 'model_weights.pt'
USE_DELTA = True
CHECKPOINTS_PATH = '.'

def get_frames(queue):
    frames = []
    for i in range(queue.qsize()):
        frame = queue.get()
        frames.append(frame)
        queue.put(frame)
    frames = torch.stack(frames).to(torch.float32)
    return frames

def dict_to_vec(dict):
    vec = []
    for k, v in dict.items():
        if type(v) == tuple:
            v1 = v[0] / 180 + 1
            v2 = v[1] / 180 + 1
            vec.append(v1)
            vec.append(v2)
        else:
            vec.append(v)

    vec = torch.tensor(vec)
    return vec

def fp_to_bin(fp):
    if fp < 0:
        return 0
    elif fp > 1:
        return 1
    else:
        return torch.round(fp).int().item()

def vec_to_dict(vec):
    dict = {
        "ESC": 0,
        "attack": fp_to_bin(vec[1]),
        "back": 0,
        "camera": (0, 0),
        "drop": fp_to_bin(vec[5]),
        "forward": 0,
        "hotbar.1": 0,
        "hotbar.2": 0,
        "hotbar.3": 0,
        "hotbar.4": 0,
        "hotbar.5": 0,
        "hotbar.6": 0,
        "hotbar.7": 0,
        "hotbar.8": 0,
        "hotbar.9": 0,
        "inventory": fp_to_bin(vec[16]),
        "jump": fp_to_bin(vec[17]),
        "left": 0,
        "pickItem": fp_to_bin(vec[19]),
        "right": 0,
        "sneak": fp_to_bin(vec[21]),
        "sprint": fp_to_bin(vec[22]),
        "swapHands": fp_to_bin(vec[23]),
        "use": fp_to_bin(vec[24]),
    }

    cam1 = torch.round((vec[3]-1)*180).int().item()
    cam2 = torch.round((vec[4]-1)*180).int().item()
    cam1 = max(min(cam1, 180), -180)
    cam2 = max(min(cam2, 180), -180)

    dict['camera'] = (cam1, cam2)

    if vec[7] > 0.5 and vec[9] < 0.5:
        dict['left'] = 1
        dict['right'] = 0
    elif vec[7] < 0.5 and vec[9] > 0.5:
        dict['left'] = 0
        dict['right'] = 1
    else:
        dict['left'] = 0
        dict['right'] = 0

    if vec[2] > 0.5 and vec[5] < 0.5:
        dict['back'] = 1
        dict['forward'] = 0
    elif vec[2] < 0.5 and vec[5] > 0.5:
        dict['back'] = 0
        dict['forward'] = 1
    else:
        dict['back'] = 0
        dict['forward'] = 0

    hotbar = vec[6:15]

    hotbar = torch.softmax(hotbar, dim=0)
    hotbar = torch.round(hotbar).int()
    for i in range(len(hotbar)):
        dict[f'hotbar.{i+1}'] = hotbar[i].item()

    return dict

def train_loop(encoder_model, rl_model, delta_model, clip, device, opt, loss_fn, episode):
    torch.cuda.empty_cache()
    transform = transforms.Compose([
        transforms.Resize((360, 640)),  # Resize images to 640x360
        transforms.ToTensor(),
    ])

    transform_clip = transforms.Compose([
        transforms.Resize((160, 256)),  # Resize images to 640x360
        transforms.ToTensor(),
    ])

    print('-'*10, f' Episode {episode} ', '-'*10)

    frame_queue = Queue(maxsize = 16)
    frame_vec_mem_queue = Queue(maxsize = 16)
    frame_vec_queue = Queue(maxsize = 16)
    action_queue = Queue(maxsize = 16)
    action_target_queue = Queue(maxsize = 16)

    env = gym.make("MineRLHumanSurvival-v0")
    env.reset()
    
    noop = {
        "ESC": 0,
        "attack": 0,
        "back": 0,
        "camera": (0, 0),
        "drop": 0,
        "forward": 0,
        "hotbar.1": 0,
        "hotbar.2": 0,
        "hotbar.3": 0,
        "hotbar.4": 0,
        "hotbar.5": 0,
        "hotbar.6": 0,
        "hotbar.7": 0,
        "hotbar.8": 0,
        "hotbar.9": 0,
        "inventory": 0,
        "jump": 0,
        "left": 0,
        "pickItem": 0,
        "right": 0,
        "sneak": 0,
        "sprint": 0,
        "swapHands": 0,
        "use": 0
    }

    for i in range(16):
        obs, _, _, _ = env.step(noop)
        obs = obs['pov']
        obs = Image.fromarray(obs, 'RGB')
        obs = transform(obs)
        obs = obs.cpu()
        frame_queue.put(obs)
        obs = obs.unsqueeze(0).to(device)
        vec = encoder_model.predict(obs).squeeze().cpu()
        frame_vec_queue.put(vec)
        if USE_DELTA:
            mem = 32 + 5 + 3
        else:
            mem = 32
        vec = torch.cat((vec, torch.zeros(mem)), dim=0)
        obs = obs.cpu()
        frame_vec_mem_queue.put(vec)
        act = torch.cat((dict_to_vec(noop), torch.zeros(32)), dim=0)
        action_queue.put(act)
        action_target_queue.put(act)

    prompts = [
        "collect dirt",
        "gather dirt",
        "mine dirt",
        "dig dirt"
    ]

    episode_loss = 0
    done = False
    success = False
    max_step = 2000
    step = 0
    with tqdm(total=max_step, desc=f"Training... ", leave=True) as pbar:
        while not done:
            torch.cuda.empty_cache()
            frame_vecs_mems = get_frames(frame_vec_mem_queue).unsqueeze(0).to(device)
            actions = get_frames(action_queue).unsqueeze(0).to(device)
            action = rl_model(frame_vecs_mems, actions).cpu()
            
            frame_vecs_mems = frame_vecs_mems.cpu().detach()
            actions = actions.cpu().detach()
            action = action[0, -1].detach()
            action_queue.get()
            action_queue.put(action.clone().requires_grad_(True))
        
            action_dict = vec_to_dict(action[:25].clone().requires_grad_(True))

            action_target_queue.get()
            action_target_queue.put(torch.cat((dict_to_vec(action_dict), action[25:].clone().requires_grad_(True)), dim=0).cpu())

            obs, _, done, info = env.step(action_dict)

            for k, v in action_dict.items():
                if k == 'error':
                    print(action_dict)
            
            obs = obs['pov']
            obs = Image.fromarray(obs, 'RGB')
            obs = transform(obs)

            frame_queue.get()
            obs = obs.cpu().detach()
            frame_queue.put(obs)
            obs = obs.to(device)
            
            obs_vec = encoder_model.predict(obs.unsqueeze(0)).squeeze().cpu().detach()
            obs = obs.cpu().detach()
            frame_vec_queue.get()
            frame_vec_queue.put(obs_vec)

            if USE_DELTA:
                frame_vecs = get_frames(frame_vec_queue).unsqueeze(0).to(device)
                tgt = get_frames(frame_vec_mem_queue)[:, 2048:2053].unsqueeze(0).to(device)
                deltas = delta_model(frame_vecs, tgt).cpu()
                frame_vecs = frame_vecs.cpu().detach()
                tgt = tgt.cpu().detach()
                deltas = deltas.squeeze()
                obs_vec = torch.cat((obs_vec, deltas[-1], action[-35:]), dim=0)
                deltas = deltas.detach()
            else:
                obs_vec = torch.cat((obs_vec, action[-32:]), dim=0)

            action = action.cpu().detach()

            frame_vec_mem_queue.get()
            frame_vec_mem_queue.put(obs_vec)
            obs_vec = obs_vec.detach()
            video = get_frames(frame_queue)
            video = video.cpu()
            v_new = torch.zeros(16, 3, 160, 256)
            for i in range(video.size(0)):
                v = video[i].numpy()
                v = Image.fromarray(v, 'RGB')
                v_new[i] = transform_clip(v)
            video = v_new
            v_new = v_new.detach()
            video = video.to(device)
            video = video.unsqueeze(0)
            image_feats = clip.forward_image_features(video)
            video = video.cpu().detach()
            video_feats = clip.forward_video_features(image_feats)
            image_feats = image_feats.cpu().detach()
            text_feats_batch = clip.encode_text(prompts)
            logits_per_video, logits_per_text = clip.forward_reward_head(
                video_feats, text_tokens=text_feats_batch
            )
            video_feats = video_feats.cpu().detach()
            text_feats_batch = text_feats_batch.cpu().detach()
            r_mean = torch.mean(logits_per_video).cpu().item()
            if r_mean > 0.95:
                success = True
                
            logits_per_text = logits_per_text.cpu().detach()
            logits_per_video = logits_per_video.cpu().detach()
            action_vecs = get_frames(action_target_queue).unsqueeze(0)
            actions = get_frames(action_queue).unsqueeze(0)

            diff = loss_fn(action_vecs, actions)
            loss = -(r_mean * 5 - torch.mean(diff * 0.001)) #-torch.mean(r_mean * torch.abs(action_vecs) - 0.001 * diff)

            action_vecs = action_vecs.cpu().detach()
            actions = actions.cpu().detach()
            diff = diff.cpu().detach()

            opt.zero_grad()
            loss.backward(retain_graph=False)
            opt.step()

            episode_loss += loss.item()

            step += 1
            pbar.update(1)
            if step >= max_step or success:
                done = True

    episode_loss /= step
    env.close()

    print('-'*10, f' End of episode {episode}, loss: {episode_loss:.4f}, steps: {step}, success: {success}', '-'*10)
    return episode_loss, success

def save_model(rl_model):
    print('Saving model...')
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    torch.save(rl_model.state_dict(), os.path.join(CHECKPOINTS_PATH, f'rl_model_delta.ckpt'))
    print('Model saved')

def fit(encoder_model, rl_model, delta_model, clip, device, opt, loss_fn, episodes):
    episode_losses = []
    successes = []
    for episode in range(episodes):
        episode_loss, success = train_loop(encoder_model, rl_model, delta_model, clip, device, opt, loss_fn, episode)
        episode_losses.append(episode_loss)
        successes.append(success)
        save_model(rl_model)
        if episode % 10 == 9:
            with open('episode_losses.txt', 'ab') as f:
                pickle.dump(episode_losses[:-10], f)
            with open('successes.txt', 'ab') as f:
                pickle.dump(successes[:-10], f)

def load_clip(checkpoint_path, device):
    clip = MineCLIP(arch="vit_base_p16_fz.v2.t2", hidden_dim=512, image_feature_dim=512, mlp_adapter_spec="v0-2.t0", pool_type="attn.d2.nh8.glusw", resolution=(160, 256)).to(device)
    return clip

def load_rl(checkpoint_path, device):
    if USE_DELTA:
        input_dim = 2048 + 5 + 32 + 3
    else:
        input_dim = 2048 + 32

    model = Transformer(
        input_dim=input_dim, feedforward_dim=2048, output_dim=(25+32), num_heads=8, num_layers=6
    ).to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = nn.HuberLoss()
    return model, opt, loss

def load_delta(checkpoint_path, device):
    model = Transformer(
        input_dim=2048, feedforward_dim=2048, output_dim=5, num_heads=8, num_layers=3
    ).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def load_encoder(checkpoint_path, device):
    model = Autoencoder().to(device)  # Initialize your model class
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def setup_models(device):
    # Load model from checkpoint
    encoder_model = load_encoder(ENCODER_PATH, device)
    rl_model, opt, loss_fn = load_rl(RL_PATH, device)
    clip = load_clip(None, device)
    if USE_DELTA:
        delta_model = load_delta(DELTA_PATH, device)
        return encoder_model, rl_model, delta_model, clip, opt, loss_fn
    return encoder_model, rl_model, None, clip, opt, loss_fn

def setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    encoder_model, rl_model, delta_model, clip, opt, loss_fn = setup_models(device)
    encoder_model.summary()
    rl_model.summary()
    if delta_model is not None:
        delta_model.summary()
    return encoder_model, rl_model, delta_model, clip, device, opt, loss_fn

# @hydra.main(config_name="conf", config_path=".", version_base="1.1")
def main():
    encoder_model, rl_model, delta_model, clip, device, opt, loss_fn = setup()
    fit(encoder_model, rl_model, delta_model, clip, device, opt, loss_fn, 10)

if __name__ == "__main__":
    main()