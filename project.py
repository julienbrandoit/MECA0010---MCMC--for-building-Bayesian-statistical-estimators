"""
This script contains the main code for the project. It includes the functions for the log posterior, the MCMC Metropolis-Hastings algorithm, the approximate Hessian matrix, the Gelman-Rubin diagnostic, and the plotting functions.
The script is divided into two parts: the first part contains the functions and the second part contains the main code.
The main code includes the plotting of the log posterior, the MCMC simulation, the convergence analysis, and the estimation of the MAP and the 95% credible region for the source location.

# Required Libraries:
- numpy
- matplotlib
- seaborn
- tqdm

The algorithmic part of the code (log posterior, MCMC, Hessian, Gelman-Rubin) was written exclusively by Brandoit Julien and Montagnino Clémence.
The reference are provided in the code where necessary [Finite difference for the Hessian and the Gelman-Rubin diagnostic].

The plotting part of the code was written by Brandoit Julien and Montagnino Clémence with the help of ChatGPT.
ChatGPT provided the fig and axs handling for the contour plot and the line plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import seaborn as sns
from tqdm import tqdm

# == Set up the plotting environment == #

sns.set_context('notebook')
sns.set_style('whitegrid')
sns.set_style('ticks')
sns_palette_heatmap = 'YlOrBr'
sns_palette_lineplot = 'Set1'
sns.set_palette(sns_palette_heatmap)

# == Set up the SEED for reproducibility == #

SEED = 5
np.random.seed(SEED)

# == Set up the parameters and the data == #

p1 = (3, 15)
p2 = (3, 16)
p3 = (4, 15)
p4 = (4, 16)
p5 = (5, 15)
p6 = (5, 16)

t1 = 3.12
t2 = 3.26
t3 = 2.98
t4 = 3.12
t5 = 2.84
t6 = 2.98

x_data = np.asarray([p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]]) # km
y_data = np.asarray([p1[1], p2[1], p3[1], p4[1], p5[1], p6[1]]) # km
t_data = np.asarray([t1, t2, t3, t4, t5, t6]) # s
data = np.array([x_data, y_data, t_data]).T # (N, 3)

v = 5 # km/s
# sigma is defined in the main code because we study both sigma = 0.1 and sigma = 0.25

# == Set up the model == #

def log_posterior(x_data, y_data, t_data, x, y, v, sigma):
    """
    Computes the log posterior probability for a set of positions (x, y) based on observed data.

    Parameters:
    -----------
    x_data : array-like
        Array of observed x-coordinates.
        
    y_data : array-like
        Array of observed y-coordinates.
        
    t_data : array-like
        Array of observed time-of-flight or distance-related data for comparison.
        
    x : array-like
        Array of x-coordinates at which the log posterior is to be evaluated.
        
    y : array-like
        Array of y-coordinates at which the log posterior is to be evaluated.
        
    v : float
        Assumed constant velocity for normalizing distances. Should be positive.
        
    sigma : float
        Uncertainty level in the observed data. Controls the spread of the Gaussian likelihood 
        and must be positive.
        
    Returns:
    --------
    p : np.ndarray
        Array of log posterior values for each (x, y) coordinate pair.
        
    Notes:
    ------
    - This function utilizes numpy's vectorization capabilities to optimize the calculation of distances 
      and log probabilities for potentially large arrays of position coordinates.
    - The uncertainty parameter, `sigma`, should be greater than zero to ensure valid Gaussian computations.
    
    """
    
    # Stack x and y coordinates into a 2D array for vectorized computation.
    xy = np.array([x, y]).T
    pos = np.array([x_data, y_data]).T
    
    # Compute Euclidean distances between each pair of (x, y) and (x_data, y_data).
    distances = np.linalg.norm(xy[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
    distances /= v  # Normalize distances by the velocity v.
    
    # Calculate log posterior based on Gaussian likelihood of distance errors.
    p = np.sum(-0.5 * ((distances - t_data) / sigma)**2, axis=1)
    
    return p

# == MCMC methods == #

def MCMC_Metropolis_Hasting(N, sigma_matrix, theta_0, sigma, burn_in_number=1000):
    """
    Perform a Metropolis-Hastings MCMC simulation to sample from the posterior distribution.

    Parameters:
    -----------
    N : int
        Number of iterations for the MCMC simulation.

    sigma_matrix : np.ndarray
        Covariance matrix for the proposal distribution. Controls the step size of the Metropolis-Hastings algorithm.

    theta_0 : list of two floats
        Initial position for the MCMC simulation.

    sigma : float
        Uncertainty level in the observed data. Controls the spread of the Gaussian likelihood 
        and must be positive.

    burn_in_number : int
        Number of burn-in iterations to discard before collecting samples.

    Returns:
    --------
    thetas : np.ndarray
        Array of shape (N - burn_in, 2) containing the posterior samples after burn-in.

    Notes:
    ------
    - The proposal distribution is a multivariate Gaussian centered at the current position.
    - The proposal distribution covariance matrix is given by `sigma_matrix`.
    - The acceptance probability is calculated based on the log posterior values at the current and proposed positions.
    - The function returns the samples after the burn-in period.
    - We try to use vectorized operations and pre-allocation of the memory as much as possible to speed up the computation.
    """
    sigma_matrix *= 2.4**2/2
    
    thetas = np.zeros((N, 2))
    thetas[0] = theta_0 
    u_rand = np.random.rand(N) # precompute the random numbers to speed up the loop
    jump = np.random.multivariate_normal([0, 0], sigma_matrix, N) # precompute the jumps to speed up the loop

    current_log_post = log_posterior(x_data, y_data, t_data, [theta_0[0]], [theta_0[1]], v, sigma)[0]
    for i in tqdm(range(N-1), desc='MCMC Simulation'):
        theta = thetas[i]
        theta_prime = theta + jump[i]
        prime_log_post = log_posterior(x_data, y_data, t_data, [theta_prime[0]], [theta_prime[1]], v, sigma)[0]

        alpha = np.exp(prime_log_post - current_log_post)

        if u_rand[i] < alpha:
            current_log_post = prime_log_post
            thetas[i+1] = theta_prime
        else:
            thetas[i+1] = theta

    return thetas[burn_in_number:]

def approximate_hessian(x, y, v, sigma):
    """
    Approximates the Hessian matrix of the log posterior at a given position (x, y).
    
    Parameters:
    -----------
    x : float
        x-coordinate at which the Hessian is to be approximated.
        
    y : float
        y-coordinate at which the Hessian is to be approximated.

    References:
    -----------
    - Yuen, Ka-Veng. (2010). Bayesian Methods for Structural Dynamics and Civil Engineering. 10.1002/9780470824566. 
    """
    h = 1e-3
    hessian = np.zeros((2, 2))

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    f_xy = -log_posterior(x_data, y_data, t_data, x, y, v, sigma)[0]

    f_x_p = -log_posterior(x_data, y_data, t_data, x + h, y, v, sigma)[0]
    f_x_m = -log_posterior(x_data, y_data, t_data, x - h, y, v, sigma)[0]

    f_y_p = -log_posterior(x_data, y_data, t_data, x, y + h, v, sigma)[0]
    f_y_m = -log_posterior(x_data, y_data, t_data, x, y - h, v, sigma)[0]

    hessian[0, 0] = (f_x_p - 2*f_xy + f_x_m) / h**2
    hessian[1, 1] = (f_y_p - 2*f_xy + f_y_m) / h**2

    return hessian


def gelman_rubin_rhat_2d(chains_2d):
    """
    Calculate the Gelman-Rubin diagnostic (R-hat) for a 2D MCMC chain.
    
    Parameters:
    - chains_2d (list of np.array): A list where each entry is an array of shape (n, 2), 
                                    containing samples from an independent MCMC chain 
                                    for (x, y) coordinates.
    
    Returns:
    - rhat_x (float): R-hat statistic for the x coordinate.
    - rhat_y (float): R-hat statistic for the y coordinate.

    References:
    ----------
    - Gelman, Andrew; Rubin, Donald B. . (1992). Inference from Iterative Simulation Using Multiple Sequences. Statistical Science, 7(4), 457–472. doi:10.1214/ss/1177011136
    """
    chains = np.array(chains_2d)
    x_chains = chains[:, :, 0]
    y_chains = chains[:, :, 1]

    # Helper function to calculate R-hat for a single dimension
    def calculate_rhat(chains):        
        n = chains.shape[1] # number of samples

        W = np.mean(np.var(chains, ddof=1, axis=1)) # within-chain variance
        B = n * np.var(np.mean(chains, axis=1), ddof=1) # between-chain variance

        if W == 0:
            W = 1e-10 # avoid division by zero when the chain is constant for the first i samples

        var_hat = ((n - 1) / n) * W + (1 / n) * B
        
        # Calculate R-hat
        rhat = np.sqrt(var_hat / W)
        return rhat
    
    # Calculate R-hat for x and y separately
    rhat_x = calculate_rhat(x_chains)
    rhat_y = calculate_rhat(y_chains)
    
    return rhat_x, rhat_y

if __name__ == '__main__':

    # ============================================== #
    # == Plot the log posterior as a contour plot == #
    # ============================================== #

    x_prior_range = (0, 20)
    y_prior_range = (0, 20)

    x_prior_range2 = (-10, 20)
    y_prior_range2 = (-5, 32.5)

    sigma1 = 0.1
    sigma2 = 0.25

    plot_res = 500
    levels_nbr = 10

    x = np.linspace(*x_prior_range, plot_res)
    y = np.linspace(*y_prior_range, plot_res)
    X, Y = np.meshgrid(x, y)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 7.5/2 + 1))

    # Plot for sigma = 0.1
    log_post1 = log_posterior(x_data, y_data, t_data, X.flatten(), Y.flatten(), v, sigma1).reshape(plot_res, plot_res)
    contour1 = axs[0].contourf(X, Y, np.exp(log_post1), alpha=0.95, levels=levels_nbr, cmap=sns_palette_heatmap)
    axs[0].scatter(x_data, y_data, color='black', marker='*', s=25, label='Seismic Stations', zorder=10)
    axs[0].set_title(r'$\sigma = {}$'.format(sigma1), fontsize=16)
    axs[0].set_xlabel(r'$x$ [km]', fontsize=14)
    axs[0].set_ylabel(r'$y$ [km]', fontsize=14)
    axs[0].set_xticks(np.arange(x_prior_range[0], x_prior_range[1] + 1, 2.5))
    axs[0].set_yticks(np.arange(y_prior_range[0], y_prior_range[1] + 1, 2.5))
    axs[0].grid(color='gray', linestyle='--', linewidth=0.5, zorder=5)

    # Plot for sigma = 0.25
    log_post2 = log_posterior(x_data, y_data, t_data, X.flatten(), Y.flatten(), v, sigma2).reshape(plot_res, plot_res)
    contour2 = axs[1].contourf(X, Y, np.exp(log_post2), alpha=0.95, levels=levels_nbr, cmap=sns_palette_heatmap)
    axs[1].scatter(x_data, y_data, color='black', marker='*', s=25, label='Seismic Stations', zorder=10)
    axs[1].set_title(r'$\sigma = {}$'.format(sigma2), fontsize=16)
    axs[1].set_xlabel(r'$x$ [km]', fontsize=14)
    axs[1].set_xticks(np.arange(x_prior_range[0], x_prior_range[1] + 1, 2.5))
    axs[1].set_yticks(np.arange(y_prior_range[0], y_prior_range[1] + 1, 2.5))
    axs[1].grid(color='gray', linestyle='--', linewidth=0.5, zorder=5)

    # Set the shared colorbar
    cbar = fig.colorbar(contour1, ax=axs, orientation='horizontal', pad=0.2, shrink=0.9, aspect=50)
    cbar.set_label(r'$\propto \pi(x, y | \mathbf{x}^{obs})$', fontsize=17)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig('log_prob_map.pdf', bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.show()

    del log_post1, log_post2
    del contour1, contour2
    del fig, axs, cbar
    del x, y, X, Y

    # =================================== #
    # == Run the first MCMC simulation == #
    # =================================== #

    # This simulation is used to approximate the MAP location of the source and the two covariance matrix for sigma = 0.1 and sigma = 0.25.
    # The initial position is sampled randomly in [0, 20]x[0, 15] and the number of iterations is set to 10000.
    # The burn-in period is set to 1000 iterations.
    # The sigma matrix is set to 1*np.eye(2) to control the step size of the Metropolis-Hastings algorithm.
    N = 100000
    burn_in = 1000

    # sigma = 0.25
    sigma_matrix2 = sigma2*np.eye(2)
    theta_0 = [np.random.uniform(0, 20), np.random.uniform(0, 15)]

    thetas = MCMC_Metropolis_Hasting(N, sigma_matrix2, theta_0, sigma2, burn_in_number=burn_in) 

    log_prob_thetas = log_posterior(x_data, y_data, t_data, thetas[:, 0], thetas[:, 1], v, sigma2)
    approximated_MAP = thetas[np.argmax(log_prob_thetas)]
    print('Approximated MAP at first run (sigma = 0.25):\n', approximated_MAP)

    x_map = approximated_MAP[0]
    y_map = approximated_MAP[1]

    hessian = approximate_hessian(x_map, y_map, v, sigma2)
    covariance2 = np.linalg.inv(hessian)
    print('Approximated Covariance matrix at MAP (sigma = 0.25):\n', covariance2)

    # sigma = 0.1
    sigma_matrix1 = sigma1*np.eye(2)
    theta_0 = [np.random.uniform(0, 20), np.random.uniform(0, 15)]

    thetas = MCMC_Metropolis_Hasting(N, sigma_matrix1, theta_0, sigma1, burn_in_number=0) # the burn-in is set to 0 to see the trajectory of the MCMC samples with the burn-in samples

    log_prob_thetas = log_posterior(x_data, y_data, t_data, thetas[:, 0], thetas[:, 1], v, sigma1)
    approximated_MAP = thetas[np.argmax(log_prob_thetas)]
    print('Approximated MAP at first run (sigma = 0.1):\n', approximated_MAP)
    
    x_map = approximated_MAP[0]
    y_map = approximated_MAP[1]

    hessian = approximate_hessian(x_map, y_map, v, sigma1)
    covariance1 = np.linalg.inv(hessian)
    print('Approximated Covariance matrix at MAP (sigma = 0.1):\n', covariance1)

    del hessian
    del theta_0, log_prob_thetas

    # == Plot the posterior samples == #
    # We use a different color for the burn-in samples and the posterior samples.

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    x_prior_range_temp = min(x_prior_range[0], np.min(thetas[:, 0]), x_prior_range[0]), max(x_prior_range[1], np.max(thetas[:, 0]), x_prior_range[1])
    y_prior_range_temp = min(y_prior_range[0], np.min(thetas[:, 1]), y_prior_range[0]), max(y_prior_range[1], np.max(thetas[:, 1]), y_prior_range[1])

    x = np.linspace(*x_prior_range_temp, plot_res)
    y = np.linspace(*y_prior_range_temp, plot_res)
    X, Y = np.meshgrid(x, y)

    log_post3 = log_posterior(x_data, y_data, t_data, X.flatten(), Y.flatten(), v, sigma1).reshape(plot_res, plot_res)

    # contour map of the posterior
    ax.contourf(X, Y, np.exp(log_post3), alpha=0.95, levels=levels_nbr, cmap=sns_palette_heatmap)

    # trajectory during burn-in
    ax.plot(thetas[:burn_in, 0], thetas[:burn_in, 1], color='red', linewidth=1, alpha=0.5, zorder=1)
    ax.scatter(thetas[:burn_in, 0], thetas[:burn_in, 1], color='red', marker='o', s=5, label='Burn-in Samples', zorder=10, alpha=0.5)

    # trajectory after burn-in
    ax.plot(thetas[burn_in:, 0], thetas[burn_in:, 1], color='blue', linewidth=0.5, alpha=0.5, zorder=1)
    ax.scatter(thetas[burn_in:, 0], thetas[burn_in:, 1], color='blue', marker='o', s=5, label='Posterior Samples', zorder=10, alpha=0.5)

    ax.scatter(x_data, y_data, color='black', marker='*', s=25, label='Seismic Stations', zorder=10)
    plt.title(r"$[\Sigma] = \mathbf{1}$", fontsize=16)
    ax.set_xlabel(r'$x$ [km]', fontsize=14)
    ax.set_ylabel(r'$y$ [km]', fontsize=14)
    ax.set_xticks(np.arange(min(2.5*(x_prior_range_temp[0]//2.5), 0), 2.5*(x_prior_range_temp[1]//2.5) + 1, 2.5))
    ax.set_yticks(np.arange(min(2.5*(y_prior_range_temp[0]//2.5), 0), 2.5*(y_prior_range_temp[1]//2.5) + 1, 2.5))
    ax.set_xlim(x_prior_range_temp)
    ax.set_ylim(y_prior_range_temp)
    ax.scatter(approximated_MAP[0], approximated_MAP[1], color='lightgreen', marker='D', s=25, label='MAP', zorder=10)
    
    ax.grid(color='gray', linestyle='--', linewidth=0.5, zorder=5)
    ax.legend(fontsize=14)

    plt.savefig('posterior_samples_first_run.png', bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.show()

    del log_post3
    del fig, ax
    del x, y, X, Y
    del x_prior_range_temp, y_prior_range_temp

    # ==================================== #
    # == Run the second MCMC simulation == #
    # ==================================== #
    
    # This simulation is used to sample from the posterior distribution.
    # The initial position is set to the MAP location obtained from the first simulation.
    # The number of iterations is set to 100000.
    # The burn-in period is set to 0 iterations.
    # The sigma matrix is set to the covariance matrix obtained from the first simulation.

    N = 1000000
    theta_0 = approximated_MAP

    # sigma = 0.1
    thetas1 = MCMC_Metropolis_Hasting(N, covariance1, theta_0, sigma1, burn_in_number=0)
    
    log_prob_thetas1 = log_posterior(x_data, y_data, t_data, thetas[:, 0], thetas[:, 1], v, sigma1)
    approximated_MAP1 = thetas[np.argmax(log_prob_thetas1)]
    print('Approximated MAP at second run (sigma = 0.1):\n', approximated_MAP1)

    # sigma = 0.25
    thetas2 = MCMC_Metropolis_Hasting(N, covariance2, theta_0, sigma2, burn_in_number=0)
    
    log_prob_thetas2 = log_posterior(x_data, y_data, t_data, thetas[:, 0], thetas[:, 1], v, sigma2)
    approximated_MAP2 = thetas[np.argmax(log_prob_thetas2)]
    print('Approximated MAP at second run (sigma = 0.25):\n', approximated_MAP2)

    # == Plot the posterior samples == #
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    x_prior_range_temp = min(x_prior_range[0], np.min(thetas1[:, 0]), x_prior_range[0]), max(x_prior_range[1], np.max(thetas1[:, 0]), x_prior_range[1])
    y_prior_range_temp = min(y_prior_range[0], np.min(thetas1[:, 1]), y_prior_range[0]), max(y_prior_range[1], np.max(thetas1[:, 1]), y_prior_range[1])

    x = np.linspace(*x_prior_range_temp, plot_res)
    y = np.linspace(*y_prior_range_temp, plot_res)
    X, Y = np.meshgrid(x, y)

    log_post3 = log_posterior(x_data, y_data, t_data, X.flatten(), Y.flatten(), v, sigma1).reshape(plot_res, plot_res)

    # contour map of the posterior
    ax.contourf(X, Y, np.exp(log_post3), alpha=0.95, levels=levels_nbr, cmap=sns_palette_heatmap)

    # trajectory (the burn-in is already burned)
    ax.plot(thetas1[:, 0], thetas1[:, 1], color='blue', linewidth=0.5, alpha=0.5, zorder=1)
    ax.scatter(thetas1[:, 0], thetas1[:, 1], color='blue', marker='o', s=5, label='Posterior Samples', zorder=10, alpha=0.5)

    ax.scatter(x_data, y_data, color='black', marker='*', s=25, label='Seismic Stations', zorder=10)
    plt.title(r"$[\Sigma] = [\Sigma]_{\sigma=0.1}$", fontsize=16)
    ax.set_xlabel(r'$x$ [km]', fontsize=14)
    ax.set_ylabel(r'$y$ [km]', fontsize=14)
    ax.set_xticks(np.arange(min(2.5*(x_prior_range_temp[0]//2.5), 0), 2.5*(x_prior_range_temp[1]//2.5) + 1, 2.5))
    ax.set_yticks(np.arange(min(2.5*(y_prior_range_temp[0]//2.5), 0), 2.5*(y_prior_range_temp[1]//2.5) + 1, 2.5))
    ax.set_xlim(x_prior_range_temp)
    ax.set_ylim(y_prior_range_temp)
    ax.scatter(approximated_MAP1[0], approximated_MAP1[1], color='lightgreen', marker='D', s=25, label='MAP', zorder=10)
    
    ax.grid(color='gray', linestyle='--', linewidth=0.5, zorder=5)
    ax.legend(fontsize=14)

    plt.savefig('posterior_samples_second_run_sigma_0.1.png', bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.show()

    del log_post3
    del fig, ax
    del x, y, X, Y
    del x_prior_range_temp, y_prior_range_temp

    # ========================== #
    # == Convergence analysis == #
    # ========================== #

    N = 25000
    num_chain = 10
    burn_in = 1000

    # sigma = 0.1
    chains = [MCMC_Metropolis_Hasting(N, covariance1, [np.random.uniform(0, 20), np.random.uniform(0, 15)], sigma1, burn_in_number=burn_in) for _ in range(num_chain)]
    
    r_x = np.zeros((chains[0].shape[0],))
    r_y = np.zeros((chains[0].shape[0],))

    for i in tqdm(range(2, r_x.shape[0]), 'Convergence Analysis, sigma=0.1'):
        r_x[i], r_y[i] = gelman_rubin_rhat_2d([chain[:i] for chain in chains])

    r_x1 = r_x[2:]
    r_y1 = r_y[2:]

    # sigma = 0.25
    chains = [MCMC_Metropolis_Hasting(N, covariance2, [np.random.uniform(0, 20), np.random.uniform(0, 15)], sigma2, burn_in_number=burn_in) for _ in range(num_chain)]

    r_x = np.zeros((chains[0].shape[0],))
    r_y = np.zeros((chains[0].shape[0],))

    for i in tqdm(range(2, r_x.shape[0]), 'Convergence Analysis, sigma=0.25'):
        r_x[i], r_y[i] = gelman_rubin_rhat_2d([chain[:i] for chain in chains])

    r_x2 = r_x[2:]
    r_y2 = r_y[2:]

    fig, ax = plt.subplots(figsize=(15, 7.5))
    colorlines = cm.get_cmap(sns_palette_lineplot, 4)
    ax.plot(np.arange(burn_in + 2, N, 1), np.log10(r_x1), label=r'$\log_{10}(\hat{R}^{\sigma=0.1}_x)$', color=colorlines(0), linewidth=2)
    ax.plot(np.arange(burn_in + 2, N, 1), np.log10(r_y1), label=r'$\log_{10}(\hat{R}^{\sigma=0.1}_y)$', color=colorlines(1), linewidth=2)
    ax.plot(np.arange(burn_in + 2, N, 1), np.log10(r_x2), label=r'$\log_{10}(\hat{R}^{\sigma=0.25}_x)$', linestyle='--', color=colorlines(2), linewidth=2)
    ax.plot(np.arange(burn_in + 2, N, 1), np.log10(r_y2), label=r'$\log_{10}(\hat{R}^{\sigma=0.25}_y)$', linestyle='--', color=colorlines(3), linewidth=2)
    ax.axhline(np.log10([1.1]), linestyle='-.', color='black', label=r'Threshold : $\log_{10}(1.1)$')

    ax.set_xlabel(r'Number of Samples [before burn-in]', fontsize=14)
    ax.set_ylabel(r'$\log_{10}(\hat{R})$', fontsize=14)
    ax.set_ylim(0, 1)
    ax.set_xlim(burn_in, N//2)
    ax.legend(fontsize=14)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, zorder=5)
    ax.set_title(r'$\sigma = 0.1 ~\text{ and } ~\sigma = 0.25$', fontsize=16)
    plt.savefig('convergence_analysis.pdf', bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.show()

    del r_x1, r_y1, r_x2, r_y2
    del fig, ax, colorlines

    # ================================================== #
    # == Estimate the MAP and the 95% credible region == #
    # ================================================== #

    # We use MCMC samples to estimate the MAP and a 95% credible REGION for the source location. 
    # For sigma = 0.1, we first build a axis-aligned credible region using the quantiles of the MCMC samples.

    alpha = 0.95
    alpha_half = (1 - alpha) / 2

    # sigma = 0.1
    x_quantiles1 = np.quantile(thetas1[:, 0], [alpha_half, 1 - alpha_half])
    y_quantiles1 = np.quantile(thetas1[:, 1], [alpha_half, 1 - alpha_half])


    x = np.linspace(*x_prior_range2, plot_res)
    y = np.linspace(*y_prior_range2, plot_res)
    X1, Y1 = np.meshgrid(x, y)
    
    log_post5 = log_posterior(x_data, y_data, t_data, X1.flatten(), Y1.flatten(), v, sigma1).reshape(plot_res, plot_res)

    # sigma = 0.25
    x_quantiles2 = np.quantile(thetas2[:, 0], [alpha_half, 1 - alpha_half])
    y_quantiles2 = np.quantile(thetas2[:, 1], [alpha_half, 1 - alpha_half])


    x = np.linspace(*x_prior_range2, plot_res)
    y = np.linspace(*y_prior_range2, plot_res)
    X2, Y2 = np.meshgrid(x, y)

    log_post6 = log_posterior(x_data, y_data, t_data, X2.flatten(), Y2.flatten(), v, sigma2).reshape(plot_res, plot_res)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 7.5 + 2))

    # Plot for sigma = 0.1
    ax = axs[0]
    contour1 = ax.contourf(X1, Y1, np.exp(log_post5), alpha=0.95, levels=levels_nbr, cmap=sns_palette_heatmap)
    ax.scatter(x_data, y_data, color='black', marker='*', s=25, label='Seismic Stations', zorder=10)
    ax.scatter(approximated_MAP[0], approximated_MAP[1], color='lightgreen', marker='D', s=25, label='MAP', zorder=10)
    ax.set_title(r'$\sigma = 0.1$', fontsize=16)
    ax.set_xlabel(r'$x$ [km]', fontsize=14)
    ax.set_ylabel(r'$y$ [km]', fontsize=14)
    ax.set_yticks(np.arange(y_prior_range2[0], y_prior_range2[1] + 1, 2.5))
    ax.set_xticks(np.arange(x_prior_range2[0], x_prior_range2[1] + 1, 2.5))
    ax.grid(color='gray', linestyle='--', linewidth=0.5, zorder=5)

    # Credible region for sigma = 0.1
    ax.plot([x_quantiles1[0], x_quantiles1[0]], [y_quantiles1[0], y_quantiles1[1]], color='black', linestyle='--', linewidth=1, label='Estimated Credible Region', zorder=10)
    ax.plot([x_quantiles1[1], x_quantiles1[1]], [y_quantiles1[0], y_quantiles1[1]], color='black', linestyle='--', linewidth=1, zorder=10)
    ax.plot([x_quantiles1[0], x_quantiles1[1]], [y_quantiles1[0], y_quantiles1[0]], color='black', linestyle='--', linewidth=1, zorder=10)
    ax.plot([x_quantiles1[0], x_quantiles1[1]], [y_quantiles1[1], y_quantiles1[1]], color='black', linestyle='--', linewidth=1, zorder=10)

    # Plot for sigma = 0.25
    ax = axs[1]
    contour2 = ax.contourf(X2, Y2, np.exp(log_post6), alpha=0.95, levels=levels_nbr, cmap=sns_palette_heatmap)
    ax.scatter(x_data, y_data, color='black', marker='*', s=25, zorder=10)
    ax.scatter(approximated_MAP[0], approximated_MAP[1], color='lightgreen', marker='D', s=25, zorder=10)
    ax.set_title(r'$\sigma = 0.25$', fontsize=16)
    ax.set_xlabel(r'$x$ [km]', fontsize=14)
    ax.set_yticks(np.arange(y_prior_range2[0], y_prior_range2[1] + 1, 2.5))
    ax.set_xticks(np.arange(x_prior_range2[0], x_prior_range2[1] + 1, 2.5))
    ax.grid(color='gray', linestyle='--', linewidth=0.5, zorder=5)

    # Credible region for sigma = 0.25
    ax.plot([x_quantiles2[0], x_quantiles2[0]], [y_quantiles2[0], y_quantiles2[1]], color='black', linestyle='--', linewidth=1, zorder=10)
    ax.plot([x_quantiles2[1], x_quantiles2[1]], [y_quantiles2[0], y_quantiles2[1]], color='black', linestyle='--', linewidth=1, zorder=10)
    ax.plot([x_quantiles2[0], x_quantiles2[1]], [y_quantiles2[0], y_quantiles2[0]], color='black', linestyle='--', linewidth=1, zorder=10)
    ax.plot([x_quantiles2[0], x_quantiles2[1]], [y_quantiles2[1], y_quantiles2[1]], color='black', linestyle='--', linewidth=1, zorder=10)

    # Create a shared legend
    handles = [Line2D([0], [0], color='black', lw=2, label=r'$C^{1D}_{\alpha = 0.95}$'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='lightgreen', markersize=10, label='Estimated MAP'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=20, label='Seismic Stations')]
    fig.legend(handles=handles, ncol=3, fontsize=18, loc='lower center', bbox_to_anchor=(0.5, 0.01))

    cbar = fig.colorbar(contour2, ax=axs, orientation='horizontal', pad=0.2, shrink=0.9, aspect=50)
    cbar.set_label(r'$\propto \pi(x, y | \mathbf{x}^{obs})$', fontsize=17)
    cbar.ax.tick_params(labelsize=14)

    # Save the figure
    plt.savefig('posterior_rectangle_credible_regions_rectangle.pdf', bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.show()

    del x_quantiles1, y_quantiles1, X1, Y1, log_post5, contour1, x_quantiles2, y_quantiles2, X2, Y2, log_post6, contour2

    # we use 2d hist to estimate the credible region that is not axis-aligned
    bins = 50 # HYPERPARAMETER

    # sigma = 0.1
    counts, x_edges, y_edges = np.histogram2d(thetas1[:, 0], thetas1[:, 1], bins=bins)

    x_size = x_edges[1] - x_edges[0]
    y_size = y_edges[1] - y_edges[0]

    sorted_counts = np.sort(counts.ravel())[::-1]
    cumsum_counts = np.cumsum(sorted_counts)
    total_counts = cumsum_counts[-1]
    threshold_idx = np.searchsorted(cumsum_counts, alpha * total_counts)
    counts_threshold = sorted_counts[threshold_idx]

    hpd_mask = counts >= counts_threshold
    
    hpd_grid = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing='ij')
    hpd_masked_values = np.zeros_like(counts)
    hpd_masked_values[hpd_mask] = counts[hpd_mask]  # Fill in only the areas above the threshold

    # Create a meshgrid for the HPD region
    hpd_x, hpd_y = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing='ij')

    hpd_x += x_size / 2
    hpd_y += y_size / 2

    hpd_x1 = hpd_x.copy()
    hpd_y1 = hpd_y.copy()
    hpd_masked_values1 = hpd_masked_values.copy()
    counts_threshold1 = counts_threshold.copy()

    del counts, x_edges, y_edges, sorted_counts, cumsum_counts, total_counts, threshold_idx, hpd_mask, hpd_grid, hpd_masked_values, hpd_x, hpd_y, x_size, y_size

    # sigma = 0.25
    counts, x_edges, y_edges = np.histogram2d(thetas2[:, 0], thetas2[:, 1], bins=bins)
    
    x_size = x_edges[1] - x_edges[0]
    y_size = y_edges[1] - y_edges[0]

    sorted_counts = np.sort(counts.ravel())[::-1]
    cumsum_counts = np.cumsum(sorted_counts)
    total_counts = cumsum_counts[-1]
    threshold_idx = np.searchsorted(cumsum_counts, alpha * total_counts)
    counts_threshold = sorted_counts[threshold_idx]

    hpd_mask = counts >= counts_threshold

    hpd_grid = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing='ij')
    hpd_masked_values = np.zeros_like(counts)
    hpd_masked_values[hpd_mask] = counts[hpd_mask]  # Fill in only the areas above the threshold

    # Create a meshgrid for the HPD region
    hpd_x, hpd_y = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing='ij')

    hpd_x += x_size / 2
    hpd_y += y_size / 2

    hpd_x2 = hpd_x.copy()
    hpd_y2 = hpd_y.copy()
    hpd_masked_values2 = hpd_masked_values.copy()
    counts_threshold2 = counts_threshold.copy()

    del counts, x_edges, y_edges, sorted_counts, cumsum_counts, total_counts, threshold_idx, hpd_mask, hpd_grid, hpd_masked_values, hpd_x, hpd_y, x_size, y_size

    # == Plot the posterior credible regions == #
    # We make 2 subplots to compare the credible regions for sigma = 0.1 and sigma = 0.25.

    x_prior_range_temp = min(x_prior_range[0], min(np.min(thetas2[:, 0]), np.min(thetas1[:,0])), x_prior_range[0]), max(x_prior_range[1], max(np.max(thetas2[:, 0]), np.max(thetas1[:,0])), x_prior_range[1])
    y_prior_range_temp = min(y_prior_range[0], min(np.min(thetas2[:, 1]), np.min(thetas1[:,1])), y_prior_range[0]), max(y_prior_range[1], max(np.max(thetas2[:, 1]), np.max(thetas1[:,1])), y_prior_range[1])

    x = np.linspace(*x_prior_range_temp, plot_res)
    y = np.linspace(*y_prior_range_temp, plot_res)
    X, Y = np.meshgrid(x, y)

        
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 7.5 + 2))

    # Sigma = 0.1

    log_post7 = log_posterior(x_data, y_data, t_data, X.flatten(), Y.flatten(), v, sigma1).reshape(plot_res, plot_res)

    ax = axs[0]
    contour_color_bar = ax.contourf(X, Y, np.exp(log_post7), alpha=0.95, levels=levels_nbr, cmap=sns_palette_heatmap)
    contour1 = ax.contour(hpd_x1, hpd_y1, hpd_masked_values1, levels=[counts_threshold1], alpha=1, zorder=10, colors='black')

    ax.scatter(x_data, y_data, color='black', marker='*', s=25, label='Seismic Stations', zorder=10)
    ax.scatter(approximated_MAP1[0], approximated_MAP1[1], color='lightgreen', marker='D', s=25, label='Estimated MAP', zorder=10)
    ax.set_title(r'$\sigma = 0.1$', fontsize=16)
    ax.set_xlabel(r'$x$ [km]', fontsize=14)
    ax.set_ylabel(r'$y$ [km]', fontsize=14)
    ax.set_xticks(np.arange(min(2.5*(x_prior_range_temp[0]//2.5), 0), 2.5*(x_prior_range_temp[1]//2.5) + 1, 2.5))
    ax.set_yticks(np.arange(min(2.5*(y_prior_range_temp[0]//2.5), 0), 2.5*(y_prior_range_temp[1]//2.5) + 1, 2.5))
    ax.set_xlim(x_prior_range_temp)
    ax.set_ylim(y_prior_range_temp)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, zorder=5)

    # Sigma = 0.25

    log_post8 = log_posterior(x_data, y_data, t_data, X.flatten(), Y.flatten(), v, sigma2).reshape(plot_res, plot_res)

    ax = axs[1]
    ax.contourf(X, Y, np.exp(log_post8), alpha=0.95, levels=levels_nbr, cmap=sns_palette_heatmap)
    contour2 = ax.contour(hpd_x2, hpd_y2, hpd_masked_values2, levels=[counts_threshold2], alpha=1, zorder=10, colors='black')

    ax.scatter(x_data, y_data, color='black', marker='*', s=25, zorder=10)
    ax.scatter(approximated_MAP2[0], approximated_MAP2[1], color='lightgreen', marker='D', s=25, zorder=10)
    ax.set_title(r'$\sigma = 0.25$', fontsize=16)
    ax.set_xlabel(r'$x$ [km]', fontsize=14)
    ax.set_xticks(np.arange(min(2.5*(x_prior_range_temp[0]//2.5), 0), 2.5*(x_prior_range_temp[1]//2.5) + 1, 2.5))
    ax.set_yticks(np.arange(min(2.5*(y_prior_range_temp[0]//2.5), 0), 2.5*(y_prior_range_temp[1]//2.5) + 1, 2.5))
    ax.set_xlim(x_prior_range_temp)
    ax.set_ylim(y_prior_range_temp)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, zorder=5)

    # Collect all legend handles and labels
    handles = [Line2D([0], [0], color='black', lw=2, label=r'$C_{\alpha = 0.95}$'), Line2D([0], [0], marker='D', color='w', markerfacecolor='lightgreen', markersize=10, label='Estimated MAP'), Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=20, label='Seismic Stations')]
    labels = [r'$C_{\alpha = 0.95}$', 'Estimated MAP', 'Seismic Stations']

    # Add the unique legend below the subplots
    fig.legend(handles=handles, labels=labels, ncol=3, fontsize=18, loc='lower center', bbox_to_anchor=(0.5, 0.01))
    
    # Set the shared colorbar
    cbar = fig.colorbar(contour_color_bar, ax=axs, orientation='horizontal', pad=0.2, shrink=0.9, aspect=50)
    cbar.set_label(r'$\propto \pi(x, y | \mathbf{x}^{obs})$', fontsize=17)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig('posterior_credible_regions.pdf', bbox_inches='tight', pad_inches=0.1, dpi=350)
    plt.show()
