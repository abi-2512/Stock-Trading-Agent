import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gym
from gym import spaces
import yfinance as yf  # For financial data API
import multiprocessing as mp

# ---------------------------------------------
# 1. Define the Stock Trading Environment
# ---------------------------------------------
class StockTradingEnv(gym.Env):
    """
    A simple stock trading environment.
    Actions:
        0: Hold
        1: Buy (buy one share)
        2: Sell (sell one share if held)
    The observation is a window (sequence) of historical data.
    """
    def __init__(self, df, initial_balance=10000, window_size=50):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_steps = len(df) - 1

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        # Observation: window_size rows of (Open, High, Low, Close, Volume)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, df.shape[1]),
            dtype=np.float32
        )
    
    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.window_size
        return self._get_observation()
    
    def _get_observation(self):
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        return np.array(obs, dtype=np.float32)
    
    def step(self, action):
        # Get the current closing price as reference
        current_price = self.df.iloc[self.current_step]['Close'].item()
        done = False
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
        
        # Update net worth and compute reward as the change in net worth
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - prev_net_worth
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
        
        next_state = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        return next_state, reward, done, {}

# ---------------------------------------------
# 2. Experience Replay Memory
# ---------------------------------------------
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# ---------------------------------------------
# 3. DQN Model with LSTM for Temporal Features
# ---------------------------------------------
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers, num_actions):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        # LSTM layer expects input shape: (batch, seq_len, input_size) with batch_first=True
        self.lstm = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, seq_len, hidden_size)
        # Use the output at the last time step
        last_out = lstm_out[:, -1, :]  # shape: (batch, hidden_size)
        out = self.fc(last_out)        # shape: (batch, num_actions)
        return out

# ---------------------------------------------
# 4. DQN Agent with Epsilon-Greedy Action Selection
# ---------------------------------------------
class DQNAgent:
    def __init__(self, state_shape, num_actions, hidden_size=64, lstm_layers=1,
                 lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        input_size = state_shape[1]  # number of features per time step
        self.model = DQN(input_size, hidden_size, lstm_layers, num_actions).to(self.device)
        self.target_model = DQN(input_size, hidden_size, lstm_layers, num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
    
    def select_action(self, state):
        # Exponential decay for epsilon
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                   np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, seq_len, input_size)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.max(1)[1].item()
    
    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
        
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # Compute current Q values
        q_values = self.model(state_batch).gather(1, action_batch)
        # Compute the target Q values using the target network
        next_q_values = self.target_model(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

# ---------------------------------------------
# 5. Training Loop
# ---------------------------------------------
def train(agent, env, num_episodes=50, batch_size=32, target_update=10):
    episode_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            agent.update(batch_size)
        episode_rewards.append(episode_reward)
        if episode % target_update == 0:
            agent.update_target()
        print(f"Episode {episode}, Total Reward: {episode_reward:.2f}")
    return episode_rewards

# ---------------------------------------------
# 6. Helper: Load Historical Stock Data Using yfinance
# ---------------------------------------------
def load_stock_data(ticker, period="5y"):
    df = yf.download(ticker, period=period)
    # Select relevant columns for the simulation
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

# ---------------------------------------------
# 7. Real-Time Inference Pipeline
# ---------------------------------------------
def real_time_inference(agent, ticker, window_size):
    """
    Fetch real-time (intraday) data for a given ticker,
    use the trained agent to decide on an action.
    """
    df = yf.download(ticker, period="1d", interval="1m")
    if len(df) < window_size:
        print(f"Not enough data for {ticker} for inference.")
        return None
    # Prepare observation: use the most recent window_size rows
    observation = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(window_size).values
    state = torch.FloatTensor(observation).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_values = agent.model(state)
    action = q_values.max(1)[1].item()
    action_map = {0: "Hold", 1: "Buy", 2: "Sell"}
    print(f"Real-time inference for {ticker}: {action_map[action]}")
    return action

# ---------------------------------------------
# 8. Parallelized Inference for Multiple Tickers
# ---------------------------------------------
def parallel_inference(agent, tickers, window_size):
    def inference_for_ticker(ticker):
        action = real_time_inference(agent, ticker, window_size)
        return (ticker, action)
    
    with mp.Pool(processes=len(tickers)) as pool:
        results = pool.map(inference_for_ticker, tickers)
    return dict(results)

# ---------------------------------------------
# 9. Main Execution: Training and Inference
# ---------------------------------------------
if __name__ == "__main__":
    # Load historical data for training (e.g., Apple Inc.)
    ticker = "AAPL"
    df = load_stock_data(ticker, period="5y")
    
    # Define window size (number of time steps fed to the LSTM)
    window_size = 50
    env = StockTradingEnv(df, initial_balance=10000, window_size=window_size)
    
    # Create the DQN agent. The state shape is (window_size, num_features)
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = DQNAgent(state_shape, num_actions)
    
    print("Starting training...")
    train(agent, env, num_episodes=50, batch_size=32, target_update=10)
    
    # After training, run a real-time inference pipeline on multiple tickers
    tickers = ["AAPL", "MSFT", "GOOG"]
    signals = parallel_inference(agent, tickers, window_size)
    print("Trading signals:", signals)
