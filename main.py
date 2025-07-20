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

# 定義 Transition 用於經驗回放
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# 貪食蛇遊戲環境
class SnakeGame:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.max_steps_without_food = 100  # 未吃食物最大步數
        self.reset()

    def reset(self):
        """重置遊戲狀態，蛇初始長度為 4"""
        self.snake = [
            (self.width // 2, self.height // 2),  # 頭部
            (self.width // 2, self.height // 2 + 1),
            (self.width // 2, self.height // 2 + 2),
            (self.width // 2, self.height // 2 + 3)
        ]  # 初始長度為 4，垂直向下
        self.food = self._place_food()
        self.direction = (0, -1)  # 初始方向：向上
        self.score = 0
        self.game_over = False
        self.steps_without_food = 0  # 未吃食物的步數計數器
        return self._get_state()

    def _place_food(self):
        """隨機放置食物，確保不在蛇身上"""
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def step(self, action):
        """執行一步動作，返回 (state, reward, done)"""
        self.steps_without_food += 1  # 每步增加計數器
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 上、右、下、左
        self.direction = directions[action]
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # 檢查撞牆或撞自己
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake):
            self.game_over = True
            return self._get_state(), -10.0, True

        self.snake.insert(0, new_head)
        reward = 0.0
        if new_head == self.food:
            self.score += 10
            self.food = self._place_food()
            reward = 10.0
            self.steps_without_food = 0  # 重置計數器
        else:
            self.snake.pop()
            # 100 步未吃到食物，給予負獎勵
            if self.steps_without_food >= self.max_steps_without_food:
                reward = -1.0
                self.steps_without_food = 0  # 重置計數器，避免連續扣分

        return self._get_state(), reward, False

    def _get_state(self):
        """獲取當前遊戲狀態（11 維向量）"""
        head = self.snake[0]
        food = self.food
        state = [
            int(head[0] == 0 or (head[0] - 1, head[1]) in self.snake),  # 左邊危險
            int(head[0] == self.width - 1 or (head[0] + 1, head[1]) in self.snake),  # 右邊危險
            int(head[1] == 0 or (head[0], head[1] - 1) in self.snake),  # 上邊危險
            int(head[1] == self.height - 1 or (head[0], head[1] + 1) in self.snake),  # 下邊危險
            int(food[0] < head[0]),  # 食物在左
            int(food[0] > head[0]),  # 食物在右
            int(food[1] < head[1]),  # 食物在上
            int(food[1] > head[1]),  # 食物在下
            int(self.direction == (0, -1)),  # 當前方向：上
            int(self.direction == (1, 0)),   # 當前方向：右
            int(self.direction == (0, 1)),   # 當前方向：下
        ]
        return np.array(state, dtype=np.float32)

    def render(self, screen=None):
        """渲染遊戲"""
        if screen is None:
            return
        screen.fill((0, 0, 0))
        cell_size = 20
        for segment in self.snake:
            pygame.draw.rect(screen, (0, 255, 0), (segment[0] * cell_size, segment[1] * cell_size, cell_size, cell_size))
        pygame.draw.rect(screen, (255, 0, 0), (self.food[0] * cell_size, self.food[1] * cell_size, cell_size, cell_size))
        pygame.display.flip()

# 經驗回放記憶體
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN 神經網絡
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN 代理
class DQNAgent:
    def __init__(self, state_dim, action_dim, device, model_path="dqn_model.pth"):
        self.device = device
        self.model_path = model_path
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayMemory(10000)
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.steps_done = 0
        self.load_model()

    def select_action(self, state):
        """選擇動作（ε-greedy）"""
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
            return
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
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        """更新目標網絡"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        """保存模型和優化器狀態"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """載入模型和優化器狀態"""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.steps_done = checkpoint['steps_done']
            print(f"Model loaded from {self.model_path}, steps_done: {self.steps_done}")
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("No saved model found, starting fresh training")

# 主訓練迴圈

def train_dqn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    env = SnakeGame()
    agent = DQNAgent(state_dim=11, action_dim=4, device=device, model_path="dqn_model.pth")
    num_episodes = 1000
    target_update = 10
    best_score = 0
    best_total_reward = float('-inf')

    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    print("Pygame initialized, training started...")

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor([state], device=device, dtype=torch.float32)
        total_reward = 0
        print(f"Episode {episode} started, initial score: {env.score}")

        while not env.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Pygame window closed, saving model and exiting...")
                    agent.save_model()
                    pygame.quit()
                    return

            action = agent.select_action(state)
            next_state, reward, done = env.step(action.item())
            total_reward += reward

            next_state = None if done else torch.tensor([next_state], device=device, dtype=torch.float32)
            reward = torch.tensor([reward], device=device)
            agent.memory.push(state, action, next_state, reward, done)

            state = next_state
            agent.optimize_model()
            env.render(screen)

            if done:
                print(f"Episode {episode} ended, Score: {env.score}, Total Reward: {total_reward}")
                break

        if episode % target_update == 0:
            agent.update_target_net()

        # 保存模型：分數提高、總獎勵提高或每 50 回合
        if env.score > best_score or total_reward > best_total_reward or episode % 50 == 0:
            if env.score > best_score:
                best_score = env.score
            if total_reward > best_total_reward:
                best_total_reward = total_reward
            agent.save_model()

    print("Training completed!")
    agent.save_model()
    pygame.quit()

if __name__ == "__main__":
    train_dqn()
