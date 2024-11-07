
import struct
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import cm
from mpld3 import fig_to_html
from io import BytesIO
from scipy.ndimage import gaussian_filter




MAX_VOL = 100

def read_data_from_binary_file(filename):
    with open(filename, "rb") as file:
        struct_size = 32
        num_options = int(os.stat(filename).st_size / struct_size) #One option correspoinds to 3 doubles
        
        maturities = []
        moneyness = []
        vols = []
        deltas = []
        print(f"Reading data for {num_options} options\n")
        for _ in range(num_options):
           # Read maturity, moneyness, and vol for each option
            option_data = file.read(struct_size)  # Assuming each option occupies 24 bytes (3 doubles)
            option_values = struct.unpack('dddd', option_data)
            if option_values[2] < MAX_VOL:
                maturities.append(option_values[0])
                moneyness.append(option_values[1])
                vols.append(option_values[2])
                deltas.append(option_values[3])
           

    return maturities, moneyness, vols, deltas


def plot_smiles():
    maturities, moneyness, vols, deltas = read_data_from_binary_file("out.bin")

    smiles = {}
    for i, T in enumerate(maturities):
        if T not in smiles.keys():
            smiles[T] = [[moneyness[i]], [vols[i]]]
        elif T in smiles.keys():
            smiles[T][0].append(moneyness[i])
            smiles[T][1].append(vols[i])
    
    # Plotting
    #fig, axes = plt.subplots(nrows=len(smiles), ncols=3, figsize=(10, 6 * len(smiles)))

    for i, (maturity, data) in enumerate(smiles.items()):
        moneyness, implied_volatility = data
        
        plt.figure()
        plt.scatter(moneyness, implied_volatility, marker='o')
        plt.title(f"Implied Volatility vs. Moneyness - {maturity}")
        plt.xlabel("Moneyness")
        plt.ylabel("Implied Volatility")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_png():
    #maturities, moneyness, vols, deltas
    x,y,z, deltas = read_data_from_binary_file("out.bin")
    
    
    y = deltas 
    x = [elem*365 for elem in x]

    xi = np.linspace(min(x), max(x), 250)
    yi = np.linspace(min(y), max(y), 250)
    xi, yi = np.meshgrid(xi, yi)

    Z = griddata((x,y), z, (xi, yi), method='linear')
    # Apply Gaussian smoothing to the Z values
    #Z_smooth = gaussian_filter(Z, sigma=0)  # You can adjust sigma for more or less smoothing

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #surf = ax.plot_surface(xi, yi, zi, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.scatter(x,y,z, color="m")
    # Plot the surface
    surf = ax.plot_surface(xi, yi, Z, cmap="jet")
    ax.plot_wireframe(xi, yi, Z, color="black", linewidth=0.5)
    ax.set_xlabel("Maturity T [days]")
    ax.set_ylabel("$\Delta$")
    ax.set_zlabel("Implied vol")
    ax.set_title("Vol surface for BTC calls (bid)")
    #Add colorbar to map colors to z values
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)    

    plt.show()

    # Send the plot image

if __name__ == '__main__':
    plot_png()

def sigma_svi(x, delta, mu, rho, omega, zeta):
    """Natural SVI parameterization with barrier penalties for 
    constraints"""
    lambd_a = 0.1
    
    rho_pen_g = (-rho + 1) #rho < 1 <-> -rho + 1 > 0
    rho_pen_l = (rho + 1) #-rho < 1 <-> rho - 1 > 0

    zeta_pen = zeta #zeta > 0
    omega_pen = omega # omega > 0

    pens = [rho_pen_g, rho_pen_l, zeta_pen, omega_pen]
    
    log_pen = sum([np.log(pen) for pen in pens])
    
    return delta + omega*0.5*(1+zeta*rho*(x - mu)+np.sqrt((zeta*(x-mu)+rho)**2 + (1-rho**2))) - lambd_a*log_pen

"""
maturities,moneyness,implied_vols, deltas = read_data_from_binary_file("out.bin")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(maturities, moneyness, implied_vols, c='r', marker='o')

ax.set_xlabel('Maturity')
ax.set_ylabel('Moneyness')
ax.set_zlabel('Implied Volatility')

plt.show()

"""

