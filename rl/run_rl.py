import minedojo
import gym

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

ENCODER_PATH = ""

def load_model_from_checkpoint(checkpoint_path):
    model = Autoencoder()  # Initialize your model class
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def setup_env():
    print("---Launching MineRL enviroment (be patient)---")
    env = HumanSurvival()
    env.reset()
    return env

def setup_models():
    # Load model from checkpoint
    encoder_model = load_model_from_checkpoint(ENCODER_PATH)
    return encoder_model

def setup():
    env = setup_env()
    encoder_model = setup_models()

def main():
    setup()

if __name__ == "__main__":
    main()