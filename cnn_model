import math
import random
import os
from collections import namedtuple, deque
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Define Transition for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Enhanced Snake Game Environment
class SnakeGame:
    def __init__(self, width=16, height=16):
        self.width = width
        self.height = height
        self.frame_stack = 10
        self.max_steps_without_food = 512  # Updated to 512
        self.state_history = deque(maxlen=self.frame_stack)
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, right, down, left
        self.reset()

    def reset(self):
        """Reset game state, snake initial length 4"""
        self.snake = [
            (self.width // 2, self.height // 2),
            (self.width // 2, self.height // 2 + 1),
            (self.width // 2, self.height // 2 + 2),
            (self.width // 2, self.height // 2 + 3)
        ]
        self.food = self._place_food()
        self.direction = (0, -1)
        self.direction_idx = 0
        self.score = 0
        self.game_over = False
        self.steps_without_food = 0
        self.steps = 0
        self.prev_dist_to_food = self._manhattan_dist(self.snake[0], self.food)
        initial_frame = self._get_single_frame()
        self.state_history = deque([initial_frame] * self.frame_stack, maxlen=self.frame_stack)
        return self._get_stacked_state()

    def _place_food(self):
        """Place food away from snake and edges with sufficient free space"""
        while True:
            food = (random.randint(2, self.width - 3), random.randint(2, self.height - 3))
            if food not in self.snake and self._manhattan_dist(food, self.snake[0]) >= 5:
                free_neighbors = sum(1 for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                                    if (0 <= food[0] + dx < self.width and
                                        0 <= food[1] + dy < self.height and
                                        (food[0] + dx, food[1] + dy) not in self.snake))
                if free_neighbors >= 2:
                    return food

    def _manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        self.steps_without_food += 1
        self.steps += 1
        invalid_action = False

        # Check for timeout (512 steps without food)
        if self.steps_without_food >= self.max_steps_without_food:
            self.game_over = True
            print("Game over due to timeout (512 steps without food)")
            return self._get_stacked_state(), -500.0, True, {'steps': self.steps, 'snake_length': len(self.snake), 'invalid_action': False, 'timeout': True}

        # Check if action leads to immediate self-collision
        new_direction = self.directions[action]
        head = self.snake[0]
        new_head = (head[0] + new_direction[0], head[1] + new_direction[1])
        if new_head in self.snake[1:]:
            invalid_action = True
            valid_actions = [i for i in range(4) if (head[0] + self.directions[i][0], head[1] + self.directions[i][1]) not in self.snake[1:]]
            if not valid_actions:
                self.game_over = True
                snake_length = len(self.snake)
                if snake_length <= 10:
                    penalty = -80.0
                elif snake_length <= 25:
                    penalty = min(-80.0 + 5.0 * (snake_length - 10), -40.0)
                else:
                    penalty = -10.0
                print("Game over due to no valid actions")
                return self._get_stacked_state(), penalty, True, {'steps': self.steps, 'snake_length': snake_length, 'invalid_action': True, 'timeout': False}
            action = random.choice(valid_actions)
            new_direction = self.directions[action]

        self.direction = new_direction
        self.direction_idx = action
        new_head = (head[0] + new_direction[0], head[1] + new_direction[1])

        # Check wall or self-collision
        snake_length = len(self.snake) + 1  # Account for new head
        if snake_length <= 10:
            wall_penalty = -100.0
            self_penalty = -80.0
        elif snake_length <= 25:
            wall_penalty = min(-100.0 + 5.0 * (snake_length - 10), -60.0)
            self_penalty = min(-80.0 + 5.0 * (snake_length - 10), -40.0)
        else:
            wall_penalty = -20.0
            self_penalty = -10.0

        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            collision_penalty = self_penalty if new_head in self.snake else wall_penalty
            if new_head[0] < 0 or new_head[0] >= self.width or new_head[1] < 0 or new_head[1] >= self.height:
                print("Game over due to wall collision")
            else:
                print("Game over due to self collision")
            return self._get_stacked_state(), collision_penalty, True, {'steps': self.steps, 'snake_length': len(self.snake), 'invalid_action': invalid_action, 'timeout': False}

        self.snake.insert(0, new_head)
        snake_length = len(self.snake)
        survival_reward = 1 if snake_length > 40 else 0.3 if snake_length > 20 else 0.1
        distance_reward = 12/snake_length
        reward = survival_reward

        new_dist = self._manhattan_dist(new_head, self.food)
        if new_dist < self.prev_dist_to_food:
            reward += distance_reward
        elif new_dist > self.prev_dist_to_food:
            reward -= distance_reward
        self.prev_dist_to_food = new_dist

        if new_head == self.food:
            self.score += 1
            if snake_length <= 20:
                reward = min(20.0 + 5.0 * (snake_length - 4), 100.0)
            elif snake_length <= 40:
                reward = min(100.0 + 10.0 * (snake_length - 20), 300.0)
            else:
                reward = min(300.0 + 20.0 * (snake_length - 40), 1000.0)
            self.food = self._place_food()
            self.steps_without_food = 0
            self.prev_dist_to_food = self._manhattan_dist(new_head, self.food)
        else:
            self.snake.pop()

        next_frame = self._get_single_frame()
        self.state_history.append(next_frame)
        return self._get_stacked_state(), reward, self.game_over, {'steps': self.steps, 'snake_length': len(self.snake), 'invalid_action': invalid_action, 'timeout': False}

    def _get_single_frame(self):
        """Return single image state (height, width, 5) with direction and distance"""
        state = np.zeros((self.height, self.width, 5), dtype=np.float32)
        head = self.snake[0]
        state[head[1], head[0], 0] = 1.0  # Head
        for segment in self.snake[1:]:
            state[segment[1], segment[0], 1] = 1.0  # Body
        state[self.food[1], self.food[0], 2] = 1.0  # Food
        state[:, :, 3] = self.direction_idx / 3.0  # Direction (0~1)
        state[:, :, 4] = self.prev_dist_to_food / (self.width + self.height) * 2.0  # Amplified distance (0~2)
        return state

    def _get_stacked_state(self):
        """Return stacked state (height, width, 5 * frame_stack)"""
        return np.concatenate(list(self.state_history), axis=-1)

    def render(self, screen=None):
        if screen is None:
            return
        screen.fill((0, 0, 0))
        cell_size = 30
        for segment in self.snake:
            pygame.draw.rect(screen, (0, 255, 0), (segment[0] * cell_size, segment[1] * cell_size, cell_size, cell_size))
        pygame.draw.rect(screen, (255, 0, 0), (self.food[0] * cell_size, self.food[1] * cell_size, cell_size, cell_size))
        pygame.display.flip()

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# CNN-based DQN Model
class DQN(nn.Module):
    def __init__(self, height, width, output_dim, frame_stack=10):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(5 * frame_stack, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * height * width, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Enhanced DQN Agent (Standard DQN)
class DQNAgent:
    def __init__(self, height, width, action_dim, device, model_path="cnn_dqn_model_3.pth", frame_stack=10):
        self.device = device
        self.model_path = model_path
        self.policy_net = DQN(height, width, action_dim, frame_stack).to(device)
        self.target_net = DQN(height, width, action_dim, frame_stack).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00005)
        self.memory = ReplayMemory(200000)
        self.batch_size = 256
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 20000  # Adjusted for balanced exploration
        self.steps_done = 0
        self.load_model()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.steps_done = checkpoint['steps_done']
                print(f"Model loaded from {self.model_path}, steps_done: {self.steps_done}")
            except RuntimeError as e:
                print(f"Error loading model due to architecture mismatch: {e}")
                print("Starting fresh training with new architecture")
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("No saved model found, starting fresh training")

# Main Training Loop
def train_dqn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    width, height = 16, 16
    env = SnakeGame(width=width, height=height)
    agent = DQNAgent(height, width, action_dim=4, device=device, model_path="cnn_dqn_model_3.pth")
    num_episodes = 1000000
    target_update = 10  # Increased frequency
    best_score = 0
    best_total_reward = float('-inf')
    writer = SummaryWriter(log_dir=f"runs/snake_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    wall_collisions = 0
    self_collisions = 0
    invalid_actions = 0
    total_episodes = 0
    long_snake_episodes = 0

    pygame.init()
    screen = pygame.display.set_mode((width * 30, height * 30))
    print("Pygame initialized, training started...")

    recent_scores = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)
    recent_steps = deque(maxlen=100)
    recent_max_lengths = deque(maxlen=100)
    recent_distances = deque(maxlen=100)

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).to(device).float()
        total_reward = 0
        max_snake_length = len(env.snake)
        total_episodes += 1
        episode_distances = []

        while not env.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Pygame window closed, saving model and exiting...")
                    agent.save_model()
                    pygame.quit()
                    writer.close()
                    return

            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action.item())
            total_reward += reward
            max_snake_length = max(max_snake_length, info['snake_length'])
            episode_distances.append(env.prev_dist_to_food)

            next_state_tensor = None if done else torch.from_numpy(next_state).permute(2, 0, 1).unsqueeze(0).to(device).float()
            reward_tensor = torch.tensor([reward], device=device)
            agent.memory.push(state, action, next_state_tensor, reward_tensor, done)

            state = next_state_tensor
            loss = agent.optimize_model()
            if loss is not None:
                writer.add_scalar("Loss/Step", loss, agent.steps_done)
            env.render(screen)

            if done:
                avg_distance = np.mean(episode_distances) if episode_distances else 0.0
                print(f"Episode {episode} ended, Score: {env.score}, Total Reward: {total_reward}, Snake Length: {info['snake_length']}, Steps: {info['steps']}")
                writer.add_scalar("Score/Episode", env.score, episode)
                writer.add_scalar("Total_Reward/Episode", total_reward, episode)
                writer.add_scalar("Snake_Length/Episode", info['snake_length'], episode)
                writer.add_scalar("Steps/Episode", info['steps'], episode)
                writer.add_scalar("Max_Snake_Length/Episode", max_snake_length, episode)
                writer.add_scalar("Distance_to_Food/Episode", avg_distance, episode)
                recent_scores.append(env.score)
                recent_rewards.append(total_reward)
                recent_steps.append(info['steps'])
                recent_max_lengths.append(max_snake_length)
                recent_distances.append(avg_distance)
                if reward == -60.0:
                    wall_collisions += 1
                elif reward == -5.0:
                    self_collisions += 1
                if info.get('invalid_action', False):
                    invalid_actions += 1
                if max_snake_length > 40:
                    long_snake_episodes += 1
                if episode % 100 == 0 and episode > 0:
                    avg_score = np.mean(recent_scores)
                    avg_reward = np.mean(recent_rewards)
                    avg_steps = np.mean(recent_steps)
                    avg_max_length = np.mean(recent_max_lengths)
                    avg_distance = np.mean(recent_distances)
                    wall_collision_rate = wall_collisions / total_episodes if total_episodes > 0 else 0
                    self_collision_rate = self_collisions / total_episodes if total_episodes > 0 else 0
                    invalid_action_rate = invalid_actions / total_episodes if total_episodes > 0 else 0
                    long_snake_rate = long_snake_episodes / total_episodes if total_episodes > 0 else 0
                    print(f"Episode {episode}, Avg Score (last 100): {avg_score:.2f}, Avg Reward (last 100): {avg_reward:.2f}, Avg Steps (last 100): {avg_steps:.2f}, Avg Max Snake Length (last 100): {avg_max_length:.2f}, Avg Distance to Food (last 100): {avg_distance:.2f}")
                    print(f"Wall Collision Rate: {wall_collision_rate:.3f}, Self Collision Rate: {self_collision_rate:.3f}, Invalid Action Rate: {invalid_action_rate:.3f}, Long Snake Rate (>40): {long_snake_rate:.3f}")
                break

        if episode % target_update == 0:
            agent.update_target_net()

        if env.score > best_score or total_reward > best_total_reward or episode % 100 == 0:
            if env.score > best_score:
                best_score = env.score
            if total_reward > best_total_reward:
                best_total_reward = total_reward
            agent.save_model()

    print("Training completed!")
    agent.save_model()
    writer.close()
    pygame.quit()

if __name__ == "__main__":
    train_dqn()
