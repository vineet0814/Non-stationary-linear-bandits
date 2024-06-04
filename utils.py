import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import scipy.signal as signal
import math
from scipy.signal import freqz
import math

def lcm_of_list(numbers):
    lcm = numbers[0]
    for i in range(1, len(numbers)):
        lcm = lcm * numbers[i] // math.gcd(lcm, numbers[i])
    return lcm

def create_dictionary(Nmax, rowSize, method):
  A = np.empty((rowSize, 0), dtype=complex)

  for N in range(1, Nmax + 1):
    c1 = np.zeros(N, dtype=complex)
    k_orig = np.arange(1, N + 1)
    k = k_orig[np.gcd(k_orig, N) == 1]

    for n in range(N):
      for a in k:
        c1[n] += np.exp(1j * 2 * np.pi * a * n / N)

    c1 = np.real(c1)

    k_orig = np.arange(1, N + 1)
    k = k_orig[np.gcd(k_orig, N) == 1]
    CN_colSize = len(k)
    CN_list = []  # List to accumulate CN_j arrays

    for j in range(CN_colSize):
      CN_j = np.roll(c1, j)
      CN_list.append(CN_j)

    CN = np.column_stack(CN_list)
    CNA = np.tile(CN, (rowSize // N, 1))
    CN_cutoff = CN[:rowSize % N, :]
    CNA = np.vstack((CNA, CN_cutoff))

    A = np.hstack((A, CNA))

  return np.round(np.real(A))




def strength_vs_period_L2(x, Pmax, method):

    Nmax = Pmax
    A = create_dictionary(Nmax, x.shape[0], method)  # Assuming you have the create_dictionary function defined

    # Penalty Vector Calculation
    penalty_vector = []
    for i in range(1, Nmax + 1):
        k = np.arange(1, i + 1)
        k_red = k[np.gcd(k, i) == 1]
        k_red_length = len(k_red)
        penalty_vector = np.concatenate((penalty_vector, i * np.ones(k_red_length)))

    penalty_vector = penalty_vector**2
    # penalty_vector = 1 + 0 * penalty_vector  # Use this if you do not want to use the penalty vector

    D = np.diag(1. / penalty_vector**2)
    PP = D @ A.T @ np.linalg.inv(A @ D @ A.T)
    s = PP @ x

    energy_s = np.zeros(Nmax)
    current_index_end = 0
    for i in range(1, Nmax + 1):
        k_orig = np.arange(1, i + 1)
        k = k_orig[np.gcd(k_orig, i) == 1]
        current_index_start = current_index_end + 1
        current_index_end = current_index_end + len(k)

        for j in range(current_index_start - 1, current_index_end):
            energy_s[i - 1] += abs(s[j])**2

    energy_s[0] = 0
    plt.figure()
    plt.stem(range(1, Nmax + 1), energy_s, linefmt='k-', basefmt=" ", markerfmt="k.")
    plt.title('l2 norm minimization')
    plt.xlabel('Period')
    plt.ylabel('Strength')
    plt.show()
    return energy_s,s

def strength_vs_period_L1(x, Pmax, method):

    Nmax = Pmax
    A = create_dictionary(Nmax, x.shape[0], method)  # Assuming you have the create_dictionary function defined

    # Penalty Vector Calculation
    penalty_vector = []
    for i in range(1, Nmax + 1):
        k = np.arange(1, i + 1)
        k_red = k[np.gcd(k, i) == 1]
        k_red_length = len(k_red)
        penalty_vector = np.concatenate((penalty_vector, i * np.ones(k_red_length)))

    penalty_vector = penalty_vector**2
    # penalty_vector = 1 + 0 * penalty_vector  # Use this if you do not want to use the penalty vector

    s = cp.Variable(A.shape[1], complex=True)
    objective = cp.Minimize(cp.norm(penalty_vector * s, 1))
    constraints = [x == A @ s]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    energy_s = np.zeros(Nmax)
    current_index_end = 0
    for i in range(1, Nmax + 1):
        k_orig = np.arange(1, i + 1)
        k = k_orig[np.gcd(k_orig, i) == 1]
        current_index_start = current_index_end + 1
        current_index_end = current_index_end + len(k)

        for j in range(current_index_start - 1, current_index_end):
            energy_s[i - 1] += abs(s.value[j])**2

    energy_s[0] = 0
    plt.figure()

    plt.stem(range(1, Nmax + 1), energy_s, linefmt='k-', basefmt=" ", markerfmt="k.")
    plt.title('l1 norm minimization')
    plt.xlabel('Period')
    plt.ylabel('Strength')
    plt.show()
    return energy_s



# Step 2: Generating a random input periodic signal

def generate_periodic_signal(Input_Periods, Input_Datalength, c ):
  L = len(Input_Periods)
  x = np.zeros(Input_Datalength)
  for i in range(L):
      if c == 1: x_temp = np.ones(5)#np.random.randn(Input_Periods[i])
      elif c == 0: x_temp = np.random.randn(Input_Periods[i])
      elif c == 2:
        if Input_Periods[i] == 5:
          x_temp = np.sin(np.array((0, 1, 2, 3, 4)) *2* np.pi/ Input_Periods[i])
        elif   Input_Periods[i] == 3:
          x_temp = np.sin(np.array((0, 1, 2)) *2* np.pi/ Input_Periods[i])
        elif   Input_Periods[i] == 2:
          x_temp = np.sin(np.array((0, 1)) *2* np.pi/ Input_Periods[i])
        elif   Input_Periods[i] == 7:
          x_temp = np.sin(np.array((0, 1, 2, 3, 4, 5, 6)) *2* np.pi/ Input_Periods[i])
        elif   Input_Periods[i] == 9:
          x_temp = np.sin(np.array((0, 1, 2, 3, 4, 5, 6, 7, 8)) *2* np.pi/ Input_Periods[i])
        elif   Input_Periods[i] == 11:
          x_temp = np.sin(np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9 , 10)) *2* np.pi/ Input_Periods[i])
      x_temp = np.tile(x_temp, int(np.ceil(Input_Datalength / Input_Periods[i])))
      x_temp = x_temp[:Input_Datalength]
      x_temp = x_temp / np.linalg.norm(x_temp, 2)
      x = x + x_temp

  x = x / np.linalg.norm(x, 2)
  return x


# Step 3: Adding Noise to the Input
def add_noise(x, Input_Datalength, SNR_dB):
  ns = np.random.randn(Input_Datalength)
  ns = ns / np.linalg.norm(ns, 2)
  Desired_Noise_Power = 10 ** (-SNR_dB / 20)
  ns = Desired_Noise_Power * ns
  x = x + ns
  return x

# Define Bandit Tracking System (our proposed method)

def bts(sims, Pmax, A, K, T, const, SNR_dB):
  Input_Datalength = 1
  average_regret = np.zeros(T-(Input_Datalength*K))
  min_regret = np.ones(T-(Input_Datalength*K)) * (T-(Input_Datalength*K))
  max_regret = np.zeros(T-(Input_Datalength*K))
  for sim in range(sims):
    print('Simulation number:', sim)
    long_seq_true = np.zeros((K,T))

    Input_Periods = [3,7,11]  # Component Periods of the input signal
    #a = generate_periodic_signal(Input_Periods, Input_Datalength, c = 2) +0.1
    long_seq_true[0,:] = generate_periodic_signal(Input_Periods, T, c = 2) + const

    Input_Periods = [9]  # Component Periods of the input signal
    #b = generate_periodic_signal(Input_Periods, Input_Datalength, c= 2)
    long_seq_true[1,:] = generate_periodic_signal(Input_Periods, T, c= 2)+const
    long_seq_noise = np.zeros((K,T))
    long_seq_noise[0,:] = add_noise(long_seq_true[0,:], T, SNR_dB)
    long_seq_noise[1,:] = add_noise(long_seq_true[1,:], T, SNR_dB)
    arm1 = A[0,:]
    arm2 = A[1,:]
    support_a = np.random.normal(0, 0.01, len(A[1,:]))
    support_b = np.random.normal(0, 0.01, len(A[1,:]))
    a1 = long_seq_noise[0,0]
    b1 = long_seq_noise[1,1]
    Nmax = Pmax
    ucb_parameter = 2
    num_pulls = np.ones(K)
    penalty_vector = []
    x = None
    y = None
    for i in range(1, Nmax + 1):
            k = np.arange(1, i + 1)
            k_red = k[np.gcd(k, i) == 1]
            k_red_length = len(k_red)
            penalty_vector = np.concatenate((penalty_vector, i * np.ones(k_red_length)))

    penalty_vector = penalty_vector**2
    #penalty_vector = 1 + 0 * penalty_vector  # Use this if you do not want to use the penalty vector
    D = np.diag(1. / penalty_vector**2)
    reward = np.zeros((K, T-(Input_Datalength*K)))
    chosen_arm_values = np.zeros( T-(Input_Datalength*K))
    for t in range(T-(Input_Datalength*K)):
        mu = [ np.array(A[(Input_Datalength*K)+t,:]) @ np.array(support_a).T,   np.array(A[(Input_Datalength*K)+t,:]) @ np.array(support_b).T]
        max_value = max(mu)  # Find the maximum value
        max_index = mu.index(max_value)
        ucb_values = [mu[0] + ucb_parameter * np.sqrt(np.log(t + 1)) / (num_pulls[0] + 1e-6), mu[1] + ucb_parameter * np.sqrt(np.log(t + 1)) /   (num_pulls[1] + 1e-6) ]
        ucb_max_value = max(ucb_values)  # Find the maximum value
        ucb_max_index = ucb_values.index(ucb_max_value)
        if ucb_max_index == 0:
          arm1 = np.vstack((arm1, A[Input_Datalength*K+t,:]))
          PP = D @ arm1.T @ np.linalg.inv(arm1 @ D @ arm1.T)
          x = np.append(a1, long_seq_noise[0,Input_Datalength*K+t])
          support_a = PP @ x
          a1 = x
          num_pulls[0] += 1
          chosen_arm_values[t] = long_seq_noise[0,(Input_Datalength*K)+t]
        elif ucb_max_index == 1:
          arm2 = np.vstack((arm2, A[Input_Datalength*K+t,:]))
          PP = D @ arm2.T @ np.linalg.inv(arm2 @ D @ arm2.T)
          y = np.append(b1, long_seq_noise[1,Input_Datalength*K+t])
          support_b = PP @ y
          b1 = y
          num_pulls[1] += 1
          chosen_arm_values[t] = long_seq_noise[1,(Input_Datalength*K)+t]
        reward[0,t] = max_index #arm index chosen
        reward[1,t] = max_value #chosen arm reward
    short_true = long_seq_true[:,Input_Datalength*K:]
    true_reward = np.max(short_true, axis = 0)
    difference = abs(true_reward - chosen_arm_values)
    cumulative_sum = np.cumsum(difference)
    average_regret += cumulative_sum
    min_regret = np.array([min(min_regret[i], cumulative_sum[i]) for i in range(T-(Input_Datalength*K))])
    max_regret = np.array([max(max_regret[i], cumulative_sum[i]) for i in range(T-(Input_Datalength*K))])
  average_regret /= sims
  return average_regret, min_regret, max_regret




