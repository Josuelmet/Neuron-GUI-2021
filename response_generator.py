# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import default_rng
from scipy.signal import square
import matplotlib.pyplot as plt

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


# Initialize some common data structures
stimulus = np.array([])
noise = np.array([])
response = np.array([])

# Random-number generator
rng = default_rng()


'''
T = total time
f = frequency
duty = duty cycle
amplitude1 = ON amplitude
amplitude2 = OFF amplitude
'''
# Function to generate a square wave
def generate_stimulus(T, dt, f, duty, amplitude1, amplitude2):
    global stimulus
    
    time = np.linspace(0, T, int(T/dt + 1))
    
    wave = square(2 * np.pi * f * time, duty=duty)
    wave += 1
    wave *= (amplitude1 - amplitude2) / 2
    wave += amplitude2
    
    # Assume for now that all stimuli are the same for each neuron
    stimulus = np.array([wave, wave, wave])
    
    plt.figure()
    plt.plot(time, stimulus.T)
    plt.title('Stimulus')
    plt.xlabel('Time')
    

# Function to generate noise
def generate_noise(T, dt, enable, mean, stddev, N=3):
    global noise, rng
    
    time = np.linspace(0, T, int(T/dt + 1))
    
    if not enable:
        noise = np.zeros((N, time.size))
        return
    
    noise = rng.normal(loc=mean,
                       scale=stddev,
                       size=time.size * N )
    
    noise = noise.reshape((3, time.size)) 
    
    


# Naka-Rushton Function
def S(x, max_rate, sigma, steepness, adaptation):
    
    const = (sigma + adaptation) ** steepness
    return (x > 0) * max_rate * (x**steepness) / (const + (x**steepness))


'''
tau = timescale
weights = weights of neurons
N = number of neurons
T = total time of simulation
dt = timestep per iteration

linear = linear vs non-linear
max_rate = maximum value of Naka-Rushton function
sigma = spread of Naka-Rushton function
steepness = steepness of Naka-Rushton function
'''
def generate_response(tau, weights, r0, N=3, T=10, dt=0.01,
                      linear=True, max_rate=100,
                      sigma=40, steepness=2,
                      use_adaptation=False, tau_adaptation=10,
                      strength_adaptation=0.7):
    
    global stimulus, response
    
    time = np.linspace(0, T, int(T/dt + 1))
    
    
    # Initial conditions
    response = np.zeros((N, len(time)))
    response[:,0] = r0
    

    # Adaptation (only relevant for non-linear)
    adaptation = np.zeros(N)
    
    
    # Iterate through time to update the responses
    if linear:
        for idx in np.arange(1, len(time)):
            previous = response[:,idx-1]
            change = np.matmul(weights - np.eye(N), previous)
            change += stimulus[:,idx] + noise[:,idx]
            response[:,idx] = previous + (dt/tau)*(change)
    else:
        for idx in np.arange(1, len(time)):
            previous = response[:,idx-1]
            
            if use_adaptation:
                dA = -adaptation + strength_adaptation*previous
                adaptation += (dt/tau_adaptation)*dA
            
            
            change = S(stimulus[:,idx] + np.matmul(weights, previous) + noise[:,idx],
                       max_rate, sigma, steepness, adaptation)
            # Inherent decay
            change -= previous
            
            response[:,idx] = previous + (dt/tau)*(change)
            
    plt.figure()
    plt.plot(time, response[0], label="Neuron 1")
    plt.plot(time, response[1], label="Neuron 2")
    plt.plot(time, response[2], label="Neuron 3")
    plt.title('Response')
    plt.xlabel('Time')
    plt.legend()
    
    
    '''
    firing_rate = 1 / (1 + np.e**-response)
    
    plt.figure()
    plt.plot(time, firing_rate[0], label="Neuron 1")
    plt.plot(time, firing_rate[1], label="Neuron 2")
    plt.plot(time, firing_rate[2], label="Neuron 3")
    plt.title('Firing Rate (Sigmoid of Response')
    plt.xlabel('Time')
    plt.legend()
    '''
    
    
    
    
    
'''
For now, assuming only 1 response wants to be tracked, r0.
For now, assuming no stimuli.
'''
'''
noises and uncertainties are in standard deviations.
'''
def kalman_filter(T, dt, measurement_std_dev, process_std_dev,
                  initial_uncertainty, initial_estimate):
    
    global rng, noise, stimulus, response
    
    # Create time array
    time = np.linspace(0, T, int(T/dt + 1))
    
    # Initialize a noise array
    measurement_noise = rng.normal(loc=0,
                                   scale=measurement_std_dev,
                                   size=time.size*3)
    measurement_noise = measurement_noise.reshape((3, time.size))
    
    # Kalman array constants
    state_transition = np.array([[1, dt],       # F
                                 [0, 1]])
    measurement_function = np.array([[1,0]])    # H
        
    '''
    control_function = np.array([0,1])          # B
    '''
    control_function = np.array([0,0])          # B

    # Apply noise to the response --> producing our observations
    observations = response[0] + measurement_noise[0]
        
        
    # Two states, one measurement
    f = KalmanFilter(dim_x = 2, dim_z = 1)
        
    f.x = np.array([initial_estimate,0]) # Initial estimate is variable,
                                         # Initial velocity is 0
        
    f.F = state_transition
        
    f.H = measurement_function
        
    f.P *= initial_uncertainty ** 2
        
    f.R = measurement_std_dev ** 2
        
    f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_std_dev**2)
    
    f.B = control_function
        
        
    # Create covariance data structure
    cov = np.zeros((len(time), 2, 2))    
    estimates = np.zeros((len(time), 2)) # There are two states
        
        
    for t in np.arange(len(time)):
        
        estimates[t] = f.x  # Store the states
        cov[t] = f.P        # Store the covariance
        
        f.predict(u=stimulus[0,t])
        f.update(z=observations[t])

        
            
    plt.figure()
    plt.plot(time, response[0], label="Truth")
    plt.plot(time, observations, ',', label="Sensor") # Use 'x' for noise illustration?
    plt.plot(time, estimates[:,0], '--', label="Estimate") # Graph position, not velocity
    plt.title('Kalman Filter Results')
    plt.ylabel('Response')
    plt.xlabel('Time')
    plt.legend()  

    # Second copy of graph with more visible noise
    plt.figure()
    plt.plot(time, response[0], label="Truth")
    plt.plot(time, observations, 'x', label="Sensor")
    plt.plot(time, estimates[:,0], '--', label="Estimate") # Graph position, not velocity
    plt.title('Kalman Filter Results')
    plt.ylabel('Response')
    plt.xlabel('Time')
    plt.legend()  
        
    '''
    Variance plotting
    plt.figure()
    plt.plot(cov[0:, 0, 0], label='Position (variance)')
    #plt.plot(cov[10:, 1, 1], label='Velocity variance')
    #plt.plot(np.abs(cov[10:, 0, 1]) + np.abs(cov[10:,1,0]), label='Summed |covariances|')
    plt.title('Uncertainty')
    plt.legend()
    '''





if __name__ == '__main__':
    
    #W = np.zeros((3,3))
    W = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0]])
    T = 10
    dt = 0.01
    time = np.linspace(0, T, int(T/dt + 1))
    
    # Generate the square-wave stimulus desired (can just be 0)
    generate_stimulus(T=T, dt=dt, f=0.1, duty=0.5,
                          amplitude1=0, amplitude2=1)
    
    generate_noise(T=T, dt=dt, enable=True, mean=0, stddev=0.5)
        
        
    # If linear response is desired, run linear response generation.
    generate_response(tau=1, weights=W, r0=np.ones(3),
                      T=T, dt=dt, linear=True)
    
    '''
    kalman_filter(T=T, dt=dt,
                  measurement_std_dev=1,
                  process_std_dev=1,
                  initial_uncertainty=1,
                  initial_estimate=0)
    '''
    
    
    '''
    Kalman filter code
    '''
    
    # Some constants
    
    MEASUREMENT_STD_DEV = 0.5
    
    measurement_noise = rng.normal(loc=0,
                                   scale=MEASUREMENT_STD_DEV,
                                   size=time.size*3)
    measurement_noise = measurement_noise.reshape((3, time.size))
    
    '''
    if False:
        
        state_transition = np.eye(6)
        state_transition[0,3] = dt
        state_transition[1,4] = dt
        state_transition[2,5] = dt
        
        measurement_function = np.zeros((3,6))      # H
        measurement_function[0,0] = 1
        measurement_function[1,1] = 1
        measurement_function[2,2] = 1
        
        # Apply noise to the observations
        measurement_noise = rng.normal(loc=0,
                                       scale=MEASUREMENT_STD_DEV,
                                       size=time.size*3)
        measurement_noise = measurement_noise.reshape((3, time.size))
        
        observations = response + measurement_noise
        
        # 6 states, 3 measurements
        f = KalmanFilter(dim_x = 6, dim_z = 3)
        
        # Zero initial position and velocity
        f.x = np.zeros(6)
        
        f.F = state_transition
        f.H = measurement_function
        
        f.P *= 50
        f.R = MEASUREMENT_STD_DEV ** 2
        
        # Approximate Q as zero except for in the derivative terms
        Q_var = 0.1
        f.Q = np.eye(6) * Q_var
        f.Q[:3, :] = np.zeros((3, 6))
     
        
        
        estimates = np.zeros((3, time.size))
        
        for t in np.arange(len(time)):
        
            estimates[:,t] = f.x[:3]
            
            f.predict()
            f.update(observations[:,t])
        
            
        plt.figure()
        plt.plot(time, observations[0], ':')
        plt.plot(time, estimates[0])
        plt.title('observations, estimates')
    '''
        
    if True:
        # Some constants
        
        state_transition = np.array([[1, dt],       # F
                                     [0, 1]])
        measurement_function = np.array([[1,0]])    # H
        
        
        '''
        Experimental (control) code:
        I think this code requires a more accurate state_transition matrix.
        '''
        control_function = np.array([0,1])        # B
        
        
        
        # Apply noise to the observed currents
        observations = response[0] + measurement_noise[0]
        
                
        
        
        # Two states, one measurement
        f = KalmanFilter(dim_x = 2, dim_z = 1)
        
        f.x = np.array([0,0]) # Zero initial position and velocity
        
        f.F = state_transition
        
        f.H = measurement_function
        
        f.P *= 50
        
        f.R = MEASUREMENT_STD_DEV ** 2
        
        f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)
        
        f.B = control_function
        
        
        # Create covariance data structure
        cov = np.zeros((len(time), 2, 2))    
        estimates = np.zeros((len(time), 2)) # There are two states
        
        
        
        for t in np.arange(len(time)):
        
            estimates[t] = f.x  # Store the states
            cov[t] = f.P        # Store the covariance
            
            '''
            Experimental (control) code:
            '''
            f.predict(u=stimulus[0,t])
            f.update(z=observations[t])
            
        
            
        
            
        plt.figure()
        plt.plot(response[0], label="Track")
        plt.plot(observations, ',', label="Sensor") # Use 'x' for noise illustration?
        plt.plot(estimates[:,0], '--', label="Estimate") # Graph position, not velocity
        plt.legend()  
        
        plt.figure()
        plt.plot(cov[1:, 0, 0], label='Position variance')
        #plt.plot(cov[10:, 1, 1], label='Velocity variance')
        #plt.plot(np.abs(cov[10:, 0, 1]) + np.abs(cov[10:,1,0]), label='Summed |covariances|')
        plt.title('Variances')
        plt.legend()
        
        '''
        from filterpy.stats import plot_covariance_ellipse
        plt.figure()
        for t in np.arange(1, len(time)):
            x = (time[t]*1000, estimates[t,0]*1000)
            P = cov[t]
            plot_covariance_ellipse(x, P)
        plt.title('ellipses')
        '''
            



    
    


'''
Kalman filter code
'''
'''
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


# Initialize data structures
rng = default_rng()


# Some constants
MEASUREMENT_VAR = 0.3
MEASUREMENT_STD_DEV = MEASUREMENT_VAR ** 0.5

state_transition = np.array([[1, dt],       # F
                             [0, 1]])
measurement_function = np.array([[1,0]])    # H



# Apply noise to the observed currents
observations = response[0] + rng.standard_normal(response[0].shape)*MEASUREMENT_STD_DEV



# Two states, one measurement
f = KalmanFilter(dim_x = 2, dim_z = 1)

f.x = np.array([0,0]) # Zero initial position and velocity

f.F = state_transition

f.H = measurement_function

f.P *= 1000

f.R = MEASUREMENT_VAR

f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1)



estimates = np.zeros(len(time))



for t in np.arange(len(time)):

    estimates[t] = f.x[0]
    
    f.predict()
    f.update(observations[t])

    

    
plt.figure()
plt.plot(response[0], label="Track")
#plt.plot(observations, ':', label="Sensor")
plt.plot(estimates, '--', label="Estimate")
plt.legend()   
'''