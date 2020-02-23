import torch
import gym
from tqdm import tqdm
from car import V
from torch.optim import Adam
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


class ProcessFrame96(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame96, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame96.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 96 * 96 * 3:
            img = np.reshape(frame, [96, 96, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (64, 85), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[11:75, :]
        x_t = np.reshape(x_t, [64, 64, 1])
        x_t = x_t / 255
        return x_t.astype(np.float32)


def plot_frame():
    idx = np.random.randint(0, frames.shape[0])
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(np.rot90(frames[idx][0].detach().cpu().numpy(), 2), cmap="gray")
    plt.axis("off")
    f.add_subplot(1, 2, 2)
    plt.imshow(np.rot90(predictions[idx][0].detach().cpu().numpy(), 2), cmap="gray")
    plt.axis("off")
    plt.show(block=True)


if __name__ == "__main__":
    dir = "/Users/andrew/PycharmProjects/HalluCar/models/"
    env = gym.make("CarRacing-v0")
    env = ProcessFrame96(env)
    num_episodes = 100000

    training_mode = True

    loss_function = torch.nn.modules.BCELoss()

    increments = [100, 200, 300, 500, 1000, 100000]
    increment_idx = 0
    MAX_MEMORY_SIZE = 50000

    embedder = V()

    try:
        1/0
        STATE_MEM = torch.load("{}/memory.pt".format(dir))
        ending_position = int(torch.load("{}/end_pos.pt".format(dir)).item())
        num_in_queue = int(torch.load("{}/num_in_queue.pt".format(dir)).item())
        embedder.load_state_dict(torch.load("{}/embedder.pt".format(dir)))

    except:
        STATE_MEM = torch.zeros(MAX_MEMORY_SIZE, 1, 64, 64)
        ending_position = 0
        num_in_queue = 0

    BATCH_SIZE = 50
    optim = Adam(embedder.parameters())

    kl_weight = 0.001

    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor(list(state)).permute(2, 0, 1).unsqueeze(0)
        frame_counter = 0
        embedder.zero_grad()
        for _ in range(increments[increment_idx]):
            frame_counter += 1
            if not training_mode:
                env.render()
            action_vector = env.action_space.sample()

            state_next, reward, terminal, _ = env.step(action_vector)

            state_next = torch.Tensor([state_next]).permute(0, 3, 1, 2)
            if frame_counter % 10 == 0:
                STATE_MEM[ending_position] = state_next
                ending_position = (ending_position + 1) % MAX_MEMORY_SIZE
                num_in_queue = min(num_in_queue + 1, MAX_MEMORY_SIZE)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            state = state_next
            if terminal:
                break
        if num_in_queue >= BATCH_SIZE:
            optim.zero_grad()
            idx = random.choices(range(num_in_queue), k=BATCH_SIZE)
            frames = STATE_MEM[idx]
            _, predictions, kl_loss = embedder(frames)
            rec_loss = loss_function(predictions, frames)
            loss = rec_loss + kl_weight*kl_loss
            print("Episode {} avg loss: {}, rec_loss: {}, weighted kl_loss: {}".format(
                ep_num, loss.item(), rec_loss, kl_loss))
            loss.backward()
            optim.step()
            plot_frame()  # Plots the actual frame and its reconstruction

            # Save STATE_MEM and model
            if ep_num % 100:
                torch.save(embedder.state_dict(), "{}/embedder.pt".format(dir))
                torch.save(STATE_MEM, "{}/memory.pt".format(dir))
                torch.save(torch.Tensor([num_in_queue]), "{}/num_in_queue.pt".format(dir))
                torch.save(torch.Tensor([ending_position]), "{}/end_pos.pt".format(dir))

        increment_idx = min(increment_idx + 1, 5)

    env.close()
