import struct
import os
import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as spopt
from mpl_toolkits.mplot3d import Axes3D

MAX_VOL = 1000

def read_data_from_binary_file(filename):
    with open(filename, "rb") as file:
        struct_size = 32
        num_options = int(os.stat(filename).st_size / struct_size)
        maturities, strikes, vols, deltas = [], [], [], []
        print(f"Reading data for {num_options} options\n")
        for _ in range(num_options):
            option_data = file.read(struct_size)
            T, K, vol, delta = struct.unpack('dddd', option_data)
            if vol < MAX_VOL:
                maturities.append(T); strikes.append(K); vols.append(vol); deltas.append(delta)
    return maturities, strikes, vols, deltas

def sabr_vol(sigma_0, alpha, beta, rho, F_0, K, T, alpha_anchor=None):
    
    if alpha_anchor:
        alpha = alpha_anchor
    
    def C(mu):  
        return mu**beta
    def D(mu): 
        return np.log((np.sqrt(1 - 2*rho*mu + mu*mu) + mu - rho) / (1 - rho))

    zeta = alpha/(sigma_0*(1-beta)) * (F_0**(1-beta) - K**(1-beta))

    epsilon = T * alpha**2

    F_mid = np.sqrt(F_0 * K)

    if K == F_0:
        return sigma_0 / (F_0**(1 - beta))

    gamma_1 = beta / F_mid
    gamma_2 = -beta*(1-beta) / (F_mid**2)

    mult = sigma_0 * C(F_mid) * np.log(F_0/K) / D(zeta) 
    term_1 = (2*gamma_2 - gamma_1**2 + 1/(F_mid**2)) / 24.0 * (sigma_0 * C(F_mid) / alpha)**2
    term_2 = (rho * gamma_1 / 4.0) * (sigma_0 * C(F_mid) / alpha)
    term_3 = (2 - 3*rho**2) / 24.0

    return mult * (1 + (term_1 + term_2 + term_3) * epsilon)

def get_underlying_price(coin: str='btc') -> float:
    with open(file=f"{coin}.bin", mode='rb') as f:
        coin_price = struct.unpack('d', f.read(8))[0]
    return coin_price

def fit_sabr(F_0: float, strikes:np.array, sigmas: np.array, T: float):
   

    def sses(params):    
        sigma_0, alpha, beta, rho = params
        sse = 0.0
        for i, K in enumerate(strikes):
            model = sabr_vol(sigma_0=sigma_0, alpha=alpha, beta=beta, rho=rho, F_0=F_0, K=K, T=T)
            sse += (model - sigmas[i])**2
        return sse
    
    sigma_0_lb = 0
    sigma_0_ub = np.inf 

    alpha_lb = 0
    alpha_ub = np.inf
    
    beta_lb = 0
    beta_ub = 0.995
    
    rho_lb = -0.990
    rho_ub = 0.990

    result = spopt.minimize(
        sses,
        x0=[0.5, 0.5, 0.5, 0.0],
        constraints=spopt.LinearConstraint(A=np.eye(4),
                                           lb=np.array([sigma_0_lb, alpha_lb, beta_lb, rho_lb]),
                                           ub=np.array([sigma_0_ub, alpha_ub, beta_ub, rho_ub]))
    )
    return result.x

def price_forward(S_0: float, T: float, r: float=0.04) -> float:
    return np.exp(-r*T) * S_0 


def main():
    maturities, strikes, vols, deltas = read_data_from_binary_file("out.bin")
    maturities = np.array(maturities, float)
    strikes    = np.array(strikes, float)
    vols       = np.array(vols, float)

    unique_T = np.unique(maturities)
    K_min, K_max = strikes.min(), strikes.max()
    K_grid = np.linspace(K_min, K_max, 80)   
    T_grid = np.array(unique_T)              
    
    IV_surface = np.zeros((len(T_grid), len(K_grid)))

    for j, T in enumerate(T_grid):
        mask = (maturities == T)
        K_obs = strikes[mask]
        IV_obs = vols[mask]

        F_0 = price_forward(get_underlying_price(), T)
        sigma_0, alpha, beta, rho = fit_sabr(F_0=F_0, strikes=K_obs, sigmas=IV_obs, T=T)

        for i, K in enumerate(K_grid):
            IV_surface[j, i] = sabr_vol(
                sigma_0=sigma_0, alpha=alpha, beta=beta, rho=rho,
                F_0=F_0, K=K, T=T
            )

    # meshgrid for plotting
    KK, TT = np.meshgrid(K_grid, T_grid*365)  # T in days

    # plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(KK, TT, IV_surface, cmap='viridis', alpha=0.85, linewidth=0)
    ax.scatter(xs=strikes, ys=[t*365 for t in maturities], zs=vols)
    
    
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (days)")
    ax.set_zlabel("Implied vol")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.title("SABR calibrated volatility surface")
    plt.show()

if __name__ == "__main__":
    main()
