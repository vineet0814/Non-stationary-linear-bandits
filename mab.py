import numpy as np

class MultiArmedBandit:
    def __init__(self, k, sigma_e,SNR, periods):
        self.T = 1000
        self.num_arms = k
        self.error_sigma = sigma_e
        self.periods = periods
        self.SNR = SNR
        self.set_setup()
    def add_noise(self, x, Input_Datalength, SNR_dB):
      ns = np.random.randn(Input_Datalength)
      ns = ns / np.linalg.norm(ns, 2)
      Desired_Noise_Power = 10 ** (-SNR_dB / 20)
      ns = Desired_Noise_Power * ns
      x = x + ns
      return x
    def set_setup(self, mode='normal', preset_mean=None):
        d = self.num_arms
        if mode == 'normal':
            arm_para = np.random.normal(0, 1, size=d)  # d vec
        else:
            arm_para = preset_mean
        self.arm_means = arm_para
    def pick_arm(self, time, values):
      arm_1 = time % self.periods[0]
      arm_2 = self.periods[0] + time % self.periods[1]
      if values[arm_1] > values[arm_2]:
        return arm_1
      return arm_2

    def run(self, runs, duration):
        self.T = duration
        self.sim_runs = runs
        self.total_regret = np.zeros((self.sim_runs, self.T))

        # Define the number of arms and their true reward probabilities
        num_simulations = self.sim_runs
        num_steps = self.T  # You can adjust the number of time steps as needed

        average_regret = np.zeros(num_steps)
        min_regret = np.ones(num_steps) * self.T
        max_regret = np.zeros(num_steps)

        for sim_num in range(num_simulations):
            print("Running Loop ", sim_num)
            # duration = 100 * (PERIODS[0] + PERIODS[1])

            num_arms = self.num_arms

            # Initialize variables to keep track of the number of pulls and estimated rewards
            num_pulls = np.zeros(num_arms)
            estimated_rewards = np.zeros(num_arms)
            true_rewards = self.arm_means

            # Initialize time step and total regret
            time_step = 0
            total_regret = 0

            # UCB parameter (you can adjust this for exploration-exploitation trade-off)
            ucb_parameter = 2.0

            # Lists to store regret and time values for plotting
            regret_values = []
            time_values = []

            # Main loop
            for time_step in range(1, num_steps+1):
                # Calculate the Upper Confidence Bound for each arm
                ucb_values = estimated_rewards + ucb_parameter * np.sqrt(np.log(time_step + 1)) / (num_pulls + 1e-6)
                # Choose the arm with the highest UCB value
                chosen_arm = self.pick_arm(time_step, ucb_values)

                # Simulate pulling the chosen arm and observing the reward
                reward = np.random.randn()*np.sqrt(self.error_sigma) + true_rewards[chosen_arm]
                reward = self.add_noise(true_rewards[chosen_arm], 1, SNR_dB = self.SNR)
                # Update the number of pulls and estimated rewards for the chosen arm
                num_pulls[chosen_arm] += 1
                estimated_rewards[chosen_arm] += (reward - estimated_rewards[chosen_arm]) / num_pulls[chosen_arm]

                # Calculate the regret and update the total regret
                best_arm = self.pick_arm(time_step, true_rewards)
                total_regret += true_rewards[best_arm] - true_rewards[chosen_arm]

                # Append values for plotting
                regret_values.append(total_regret)
                time_values.append(time_step)

            average_regret += regret_values
            min_regret = np.array([min(min_regret[i], regret_values[i]) for i in range(num_steps)])
            max_regret = np.array([max(max_regret[i], regret_values[i]) for i in range(num_steps)])

        average_regret /= num_simulations
        # Plotting the regret as a function of time
        '''
        plt.figure(1, figsize=(10, 10))
        plt.plot(time_values, average_regret, label="MAB-UCB")
        plt.fill_between(time_values, min_regret, max_regret, alpha=0.3)
        plt.xlabel("Time Step")
        plt.ylabel("Regret")
        plt.title("Monte Carlo Regret of UCB Algorithms on Periodic Bandits")
        plt.legend()
        plt.grid()
        plt.show()
        '''
        return average_regret, time_values, min_regret, max_regret
