import numpy as np
import matplotlib.pyplot as plt
from math import tan, radians

mu0 = 4*np.pi*1e-7

def deg_to_rad(deg):
    return radians(deg)

# ==========================================================
# Prism field helpers (cube field)
# NOTE: keep scalars/arrays compatible (np works on arrays)
# ==========================================================
def F2_fn(x, y, z, a, b, c):
    R1 = np.sqrt((x + a)**2 + (y - b)**2 + (z + c)**2)
    R2 = np.sqrt((x + a)**2 + (y + b)**2 + (z + c)**2)
    return (R1 + b - y) / (R2 - b - y + 1e-15)

def F1_fn(x, y, z, a, b, c):
    R = np.sqrt((x + a)**2 + (y + b)**2 + (z + c)**2)
    return np.arctan2((x + a)*(y + b), (z + c)*R + 1e-15)

def Bz_single_prism(x, y, z, a, b, c, M):
    # Works with x,y,z as scalars or numpy arrays
    S = (
        F1_fn(-x,  y,  z, a,b,c) + F1_fn(-x,  y, -z, a,b,c) +
        F1_fn(-x, -y,  z, a,b,c) + F1_fn(-x, -y, -z, a,b,c) +
        F1_fn( x,  y,  z, a,b,c) + F1_fn( x,  y, -z, a,b,c) +
        F1_fn( x, -y,  z, a,b,c) + F1_fn( x, -y, -z, a,b,c)
    )
    return -(mu0 * M / (4*np.pi)) * S

# ==========================================================
# Cylinder quadrature points (precompute once)
# ==========================================================
def cylinder_quadrature_points(r_cyl, L_cyl, Nr=8, Ntheta=20, Nz=4):
    """
    Midpoint-rule discretization of cylinder volume.
    Returns y_off, z_off, dV arrays (flattened).
    Cylinder centered at origin, axis along z.
    """
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
    dV = (RHO * DR * DTH * DZ).ravel()  # cylindrical Jacobian rho

    return y_off, z_off, dV

# ==========================================================
# Vectorized array field (sum magnets, but y/z arrays at once)
# ==========================================================
def Bz_total_array_vec(x, y_arr, z_arr, a,b,c, Mmag, num_magnets, step_x, step_y):
    """
    x can be scalar; y_arr,z_arr are arrays (same shape).
    Returns array of Bz at those points.
    """
    Bz = np.zeros_like(y_arr, dtype=float)
    for i in range(num_magnets):
        xr = x - i*step_x
        yr = y_arr - i*step_y
        Bz += Bz_single_prism(xr, yr, z_arr, a,b,c, Mmag)
    return Bz

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    # --- cube magnets
    a = b = c = 0.005
    Mmag = 1.05e6
    num_magnets = 4
    step_y = 0.022
    step_x = tan(deg_to_rad(13)) * step_y  # 13° angle between magnets
    print(step_x)
    

    # --- cylinder (your values)
    r_cyl = 0.0175
    L_cyl = 0.004


    # --- magnetization
    M_cyl = 1.21/mu0  # A/m (steel saturation)

    # --- vehicle mass
    M_vehicle = 0.07125  # kg

    # --- scan path (center of cylinder)
    x_fixed = -0.052
    z0 = 0.0175
    y_vals = np.linspace(-0.01, 0.08, 250)

    # --- derivative step
    h = 5e-4

    # --- precompute cylinder volume points ONCE (speed!)
    # tune resolution here: lower = faster, higher = more accurate
    Nr, Ntheta, Nz = 8, 20, 4
    y_off, z_off, dV = cylinder_quadrature_points(r_cyl, L_cyl, Nr=Nr, Ntheta=Ntheta, Nz=Nz)

    # ==========================================================
    # FAST finite-cylinder force Fy(y0)
    # Fy = ∫ M_cyl * (∂Bz/∂y) dV   (magnetization along z)
    # ==========================================================
    def Fy_cylinder_fast(x_cyl, y_cyl, z_cyl, k_steel=1.0):
        y_global = y_cyl + y_off
        z_global = z_cyl + z_off

        # central-difference derivative (faster than 5-point, usually enough)
        B_plus  = Bz_total_array_vec(x_cyl, y_global + h, z_global,
                                     a,b,c, Mmag, num_magnets, step_x, step_y)
        B_minus = Bz_total_array_vec(x_cyl, y_global - h, z_global,
                                     a,b,c, Mmag, num_magnets, step_x, step_y)

        dBz_dy = k_steel * (B_plus - B_minus) / (2*h)
        Fy = np.sum(M_cyl * dBz_dy * dV)
        return Fy

    # ==========================================================
    # Acceleration curve and average via integration
    # ==========================================================
    def ay_curve_fast(k_steel):
        ay = np.zeros_like(y_vals, dtype=float)
        for i, y0 in enumerate(y_vals):
            Fy = Fy_cylinder_fast(x_fixed, y0, z0, k_steel=k_steel)
            ay[i] = Fy / M_vehicle
        return ay

    def avg_integral_accel(ay):
        return np.trapz(ay, y_vals) / (y_vals[-1] - y_vals[0])

   # ==========================================================
    # ANGLE SWEEP (7° to 15°)
    # ==========================================================

    k_steel = 1.6

    angles_deg = np.linspace(0, 90, 17)
    avg_accels = []

    for angle in angles_deg:

        # update geometry
        step_x = np.tan(np.deg2rad(angle)) * step_y

        # compute acceleration curve for this angle
        ay_vals = np.zeros_like(y_vals)

        for i, y0 in enumerate(y_vals):
            Fy = Fy_cylinder_fast(
                x_fixed,
                y0,
                z0,
                k_steel=k_steel
            )
            ay_vals[i] = Fy / M_vehicle

        # compute average acceleration
        a_avg = np.trapz(ay_vals, y_vals) / (y_vals[-1] - y_vals[0])
        avg_accels.append(a_avg)

        print(f"Angle = {90-angle:.1f}°  ->  Avg acceleration = {a_avg:.4f} m/s²")


    # ==========================================================
    # Plot angle dependence
    # ==========================================================

    plt.figure(figsize=(10,5))
    plt.plot(angles_deg, avg_accels, marker='o')
    plt.xlabel("Plate angle (deg)")
    plt.ylabel("Average acceleration (m/s²)")
    plt.title("Average acceleration vs Plate Angle (k_steel = 1.5)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()