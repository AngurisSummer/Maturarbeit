import numpy as np
import matplotlib.pyplot as plt
from math import tan, radians
from scipy.interpolate import BarycentricInterpolator

mu0 = 4*np.pi*1e-7

# ==========================================================
# Prism field helpers (cube field)
# ==========================================================
def F1_fn(x, y, z, a, b, c):
    R = np.sqrt((x + a)**2 + (y + b)**2 + (z + c)**2)
    return np.arctan2((x + a)*(y + b), (z + c)*R + 1e-15)

def Bz_single_prism(x, y, z, a, b, c, M):
    S = (
        F1_fn(-x,  y,  z, a,b,c) + F1_fn(-x,  y, -z, a,b,c) +
        F1_fn(-x, -y,  z, a,b,c) + F1_fn(-x, -y, -z, a,b,c) +
        F1_fn( x,  y,  z, a,b,c) + F1_fn( x,  y, -z, a,b,c) +
        F1_fn( x, -y,  z, a,b,c) + F1_fn( x, -y, -z, a,b,c)
    )
    return -(mu0 * M / (4*np.pi)) * S

def cylinder_quadrature_points(r_cyl, L_cyl, Nr=8, Ntheta=20, Nz=4):
    rho_edges = np.linspace(0.0, r_cyl, Nr+1)
    rho = 0.5*(rho_edges[:-1] + rho_edges[1:])
    dr = rho_edges[1:] - rho_edges[:-1]
    theta = np.linspace(0.0, 2*np.pi, Ntheta, endpoint=False)
    dtheta = 2*np.pi / Ntheta
    z_edges = np.linspace(-L_cyl/2, L_cyl/2, Nz+1)
    zc = 0.5*(z_edges[:-1] + z_edges[1:])
    dz = z_edges[1:] - z_edges[:-1]
    RHO, TH, ZC = np.meshgrid(rho, theta, zc, indexing="ij")
    DR, DTH, DZ = np.meshgrid(dr, np.full_like(theta, dtheta), dz, indexing="ij")
    y_off = (RHO * np.cos(TH)).ravel()
    z_off = (ZC).ravel()
    dV = (RHO * DR * DTH * DZ).ravel()
    return y_off, z_off, dV

def Bz_total_array_vec(x, y_arr, z_arr, a,b,c, Mmag, num_magnets, step_x, step_y):
    Bz = np.zeros_like(y_arr, dtype=float)
    for i in range(num_magnets):
        xr = x - i*step_x
        yr = y_arr - i*step_y
        Bz += Bz_single_prism(xr, yr, z_arr, a,b,c, Mmag)
    return Bz

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    # Magnet/Vehicle Params
    a = b = c = 0.005
    Mmag = 1.05e6
    step_y = 0.022
    step_x = tan(radians(13)) * step_y
    r_cyl, L_cyl = 0.0175, 0.004
    M_cyl = 1.21/mu0
    M_vehicle = 0.07125
    x_fixed, z0 = -0.052, 0.0175
    h = 5e-4
    k_steel = 1.13

    y_off, z_off, dV = cylinder_quadrature_points(r_cyl, L_cyl)

    magnet_counts = np.arange(1, 16) # Sweeping up to 15 magnets
    avg_accels = []

    print("Starting Simulation...")
    for nm in magnet_counts:
        # DYNAMIC WINDOW: Window length scales with magnet count
        # This prevents the 'decreasing' trend caused by fixed-length dilution
        y_max_window = (nm - 1) * step_y + 0.05
        y_vals_dynamic = np.linspace(-0.05, y_max_window, 200)
        
        ay_vals = np.zeros_like(y_vals_dynamic)
        for i, y0 in enumerate(y_vals_dynamic):
            # Calculate Force Fy
            y_glob = y0 + y_off
            z_glob = z0 + z_off
            B_plus  = Bz_total_array_vec(x_fixed, y_glob + h, z_glob, a,b,c, Mmag, nm, step_x, step_y)
            B_minus = Bz_total_array_vec(x_fixed, y_glob - h, z_glob, a,b,c, Mmag, nm, step_x, step_y)
            
            dBz_dy = k_steel * (B_plus - B_minus) / (2*h)
            Fy = np.sum(M_cyl * dBz_dy * dV)
            ay_vals[i] = Fy / M_vehicle
        
        # Integrate over the dynamic window
        a_avg = np.trapz(ay_vals, y_vals_dynamic) / (y_vals_dynamic[-1] - y_vals_dynamic[0])
        avg_accels.append(a_avg)
        print(f"Magnets: {nm} | Avg Accel: {a_avg:.4f} m/s²")

    # ==========================================================
    # LAGRANGE INTERPOLATION & MAXIMA SEARCH
    # ==========================================================
    # Interpolate using Barycentric for stability
    poly = BarycentricInterpolator(magnet_counts, avg_accels)
    
    x_high_res = np.linspace(1, 15, 1000)
    y_poly = poly(x_high_res)

    # Numerical derivatives for finding the maxima
    dx = x_high_res[1] - x_high_res[0]
    f_prime = np.gradient(y_poly, dx)
    f_double_prime = np.gradient(f_prime, dx)

    # Search for Maxima: f' changes from + to - and f'' < 0
    maxima_indices = np.where((np.diff(np.sign(f_prime)) < 0) & (f_double_prime[:-1] < 0))[0]

    # ==========================================================
    # PLOTTING
    # ==========================================================
    plt.figure(figsize=(10, 6))
    plt.plot(magnet_counts, avg_accels, 'ro', label='Simulation Data')
    plt.plot(x_high_res, y_poly, 'b-', alpha=0.6, label='Lagrange Interpolation')

    if len(maxima_indices) > 0:
        idx = maxima_indices[0]
        plt.annotate(f'Peak: {y_poly[idx]:.3f} m/s²\nMagnets: {x_high_res[idx]:.2f}', 
                     xy=(x_high_res[idx], y_poly[idx]), xytext=(x_high_res[idx]+1, y_poly[idx]),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
        plt.plot(x_high_res[idx], y_poly[idx], 'gs', markersize=8, label='Theoretical Maxima')

    plt.title("Magnetic Acceleration vs. Number of Magnets")
    plt.xlabel("Number of Magnets")
    plt.ylabel("Average Acceleration (m/s²)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()