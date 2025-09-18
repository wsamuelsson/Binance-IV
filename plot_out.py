import struct
import os
import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as spopt
from mpl_toolkits.mplot3d import Axes3D

MAX_VOL = 0.9

def read_data_from_binary_file(filename, underlying=None):
    with open(filename, "rb") as file:
        struct_size = 32
        num_options = int(os.stat(filename).st_size / struct_size)
        maturities, strikes, vols, deltas = [], [], [], []
        print(f"Reading data for {num_options} options\n")
        for _ in range(num_options):
            option_data = file.read(struct_size)
            T, K, vol, delta = struct.unpack('dddd', option_data)
            if (vol < MAX_VOL):
                if underlying:
                    if (K > 0.5*underlying) and (K < 1.5*underlying):
                        maturities.append(T); strikes.append(K); vols.append(vol); deltas.append(delta)
                else:
                    maturities.append(T); strikes.append(K); vols.append(vol); deltas.append(delta)
    return maturities, strikes, vols, deltas

def sabr_vol(sigma_0, alpha, rho, F_0, K, T,beta = 0.2):
    
    
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

def fit_sabr(F_0: float, strikes:np.array, sigmas: np.array, T: float,beta=0.2):
   
    # find nearest-to-ATM strike
    i_atm = np.argmin(np.abs(strikes - F_0))
    sigma_atm = sigmas[i_atm]
    
    def sses(params):
        mu = 15
        sigma_0, alpha, rho = params
        sse = 0.0
        for i, K in enumerate(strikes):
            model = sabr_vol(sigma_0=sigma_0, alpha=alpha, rho=rho, F_0=F_0, K=K, T=T)
            sse += (model - sigmas[i])**2
        #Anchor loss at atm vol
        return sse + mu * (sigma_atm - sabr_vol(sigma_0=sigma_0, alpha=alpha, rho=rho, F_0=F_0, K=strikes[i_atm], T=T))**2
    
    sigma_0_lb = 0
    sigma_0_ub = np.inf 

    alpha_lb = 0
    alpha_ub = np.inf
    
    rho_lb = -0.990
    rho_ub = 0.990

    result = spopt.minimize(
        sses,
        x0=[0.5, 0.5, 0.1],
        constraints=spopt.LinearConstraint(A=np.eye(3),
                                           lb=np.array([sigma_0_lb, alpha_lb, rho_lb]),
                                           ub=np.array([sigma_0_ub, alpha_ub, rho_ub]))
    )
    
    return result.x

def price_forward(S_0: float, T: float, r: float=0.04) -> float:
    return np.exp(r*T) * S_0 


def main():
    underlying = get_underlying_price()

    # Load calls and puts
    maturities_calls, strikes_calls, vols_calls, deltas_calls = read_data_from_binary_file("out_c.bin", underlying=underlying)
    maturities_puts,  strikes_puts,  vols_puts,  deltas_puts  = read_data_from_binary_file("out_p.bin", underlying=underlying)
    print(len(maturities_puts))
    print(len(maturities_calls))
    # Convert to arrays
    maturities_calls = np.array(maturities_calls, float)
    strikes_calls    = np.array(strikes_calls, float)
    vols_calls       = np.array(vols_calls, float)

    maturities_puts  = np.array(maturities_puts, float)
    strikes_puts     = np.array(strikes_puts, float)
    vols_puts        = np.array(vols_puts, float)

    unique_T = np.unique(np.round(np.concatenate([maturities_calls, maturities_puts]), 6))
    
    K_min = min(strikes_calls.min(), strikes_puts.min())
    K_max = max(strikes_calls.max(), strikes_puts.max())
    K_grid = np.linspace(K_min, K_max, 80)
    T_grid = np.array(unique_T)
    
    IV_surface = np.zeros((len(T_grid), len(K_grid)))
    tol = 0.5/365  # tolerance for matching maturities
    
    for j, T in enumerate(T_grid):
        F_0 = price_forward(underlying, T)


        mask_calls = (np.isclose(maturities_calls, T, atol=tol)) & (strikes_calls >= F_0)
        mask_puts  = (np.isclose(maturities_puts,  T, atol=tol)) & (strikes_puts  <= F_0)

        K_obs  = np.concatenate([strikes_puts[mask_puts], strikes_calls[mask_calls]])
        IV_obs = np.concatenate([vols_puts[mask_puts],   vols_calls[mask_calls]])

        K_obs = np.concatenate([strikes_puts[mask_puts], strikes_calls[mask_calls]])
        IV_obs = np.concatenate([vols_puts[mask_puts], vols_calls[mask_calls]])

        

    
        sigma_0, alpha, rho = fit_sabr(F_0=F_0, strikes=K_obs, sigmas=IV_obs, T=T)

        for i, K in enumerate(K_grid):
            IV_surface[j, i] = sabr_vol(
                sigma_0=sigma_0, alpha=alpha, rho=rho,
                F_0=F_0, K=K, T=T
            )

        # Plot the shortest maturity smile
        if np.isclose(T, T_grid.min(), atol=tol) or np.isclose(T, T_grid[1], atol=tol) or np.isclose(T, sorted(T_grid)[len(T_grid) // 2], atol=tol) :
            plt.figure()
            K_min_grid = np.linspace(np.min(K_obs), np.max(K_obs), 100)
            iv_min_surf = [sabr_vol(sigma_0=sigma_0, alpha=alpha, rho=rho,
                                    F_0=F_0, K=K, T=T) for K in K_min_grid]
            plt.plot(K_min_grid, iv_min_surf, label="SABR fit")
            plt.scatter(strikes_calls[mask_calls], vols_calls[mask_calls], label="Market calls")
            plt.scatter(strikes_puts[mask_puts], vols_puts[mask_puts], label="Market puts")
            plt.axvline(F_0, ls="dashed", c="red", label="Forward")
            plt.title(f"{round(T*365, 2)} dte SABR vol smile (β=0.2)")
            plt.legend()
            plt.grid(True)

    # Meshgrid for surface
    KK, TT = np.meshgrid(K_grid, T_grid*365)

    # Plot surface
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(KK, TT, IV_surface, cmap="RdYlBu", alpha=0.85, linewidth=0)

    # Scatter raw data (both puts and calls)
    ax.scatter(xs=strikes_calls, ys=maturities_calls*365, zs=vols_calls, c="blue", marker="o", label="Calls")
    ax.scatter(xs=strikes_puts,  ys=maturities_puts*365,  zs=vols_puts,  c="green", marker="^", label="Puts")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (days)")
    ax.set_zlabel("Implied vol")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.title("SABR calibrated volatility surface")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
