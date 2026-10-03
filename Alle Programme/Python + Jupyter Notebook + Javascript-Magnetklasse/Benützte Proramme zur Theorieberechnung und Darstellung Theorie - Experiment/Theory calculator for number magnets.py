import numpy as np
import matplotlib.pyplot as plt

mu0 = 4 * np.pi * 1e-7


# ==========================================================
# Prism field helpers
# ==========================================================

def F1_fn(x, y, z, a, b, c):
    R = np.sqrt(
        (x + a)**2 +
        (y + b)**2 +
        (z + c)**2
    )

    return np.arctan2(
        (x + a) * (y + b),
        (z + c) * R + 1e-15
    )


def Bz_single_prism(x, y, z, a, b, c, M):

    S = (
        F1_fn(-x,  y,  z, a, b, c)
        + F1_fn(-x,  y, -z, a, b, c)
        + F1_fn(-x, -y,  z, a, b, c)
        + F1_fn(-x, -y, -z, a, b, c)
        + F1_fn( x,  y,  z, a, b, c)
        + F1_fn( x,  y, -z, a, b, c)
        + F1_fn( x, -y,  z, a, b, c)
        + F1_fn( x, -y, -z, a, b, c)
    )

    return -(mu0 * M / (4 * np.pi)) * S


# ==========================================================
# Cylinder quadrature
# ==========================================================

def cylinder_quadrature_points(
    r_cyl,
    L_cyl,
    Nr=8,
    Ntheta=20,
    Nz=4
):

    rho_edges = np.linspace(0.0, r_cyl, Nr + 1)
    rho = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    dr = rho_edges[1:] - rho_edges[:-1]

    theta = np.linspace(
        0.0,
        2 * np.pi,
        Ntheta,
        endpoint=False
    )

    dtheta = 2 * np.pi / Ntheta

    z_edges = np.linspace(
        -L_cyl / 2,
        L_cyl / 2,
        Nz + 1
    )

    zc = 0.5 * (
        z_edges[:-1] +
        z_edges[1:]
    )

    dz = (
        z_edges[1:] -
        z_edges[:-1]
    )

    RHO, TH, ZC = np.meshgrid(
        rho,
        theta,
        zc,
        indexing="ij"
    )

    DR, DTH, DZ = np.meshgrid(
        dr,
        np.full_like(theta, dtheta),
        dz,
        indexing="ij"
    )

    y_off = (
        RHO * np.cos(TH)
    ).ravel()

    z_off = ZC.ravel()

    dV = (
        RHO *
        DR *
        DTH *
        DZ
    ).ravel()

    return y_off, z_off, dV


# ==========================================================
# Field from one row of cube magnets
# ==========================================================

def Bz_total_array_vec(
    x,
    y_arr,
    z_arr,
    a,
    b,
    c,
    Mmag,
    num_magnets,
    step_x,
    step_y
):

    Bz = np.zeros_like(
        y_arr,
        dtype=float
    )

    for i in range(num_magnets):

        xr = x - i * step_x
        yr = y_arr - i * step_y

        Bz += Bz_single_prism(
            xr,
            yr,
            z_arr,
            a,
            b,
            c,
            Mmag
        )

    return Bz


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # Cube magnets
    # ======================================================

    a = b = c = 0.005
    Mmag = 1.05e6
    num_magnets = 4


    # ======================================================
    # Moving cylindrical magnet
    # ======================================================

    r_cyl = 0.0175
    L_cyl = 0.004

    Br_cyl = 1.21
    M_cyl = Br_cyl / mu0


    # ======================================================
    # Effective inertial mass
    # ======================================================

    M_physical = 0.07125



    M_effective = (
        M_physical
        +
        0.027
    )

    print(
        "Effective mass =",
        M_effective,
        "kg"
    )


    # ======================================================
    # Geometry / scan interval
    # ======================================================

    z0 = 0.0175

    y_vals = np.linspace(
        0.025,
        0.22,
        250
    )

    # measured lateral distances for the two sides
    x_fixed_77 = +0.056

    # IMPORTANT:
    # replace this with the actual measured value
    # for the 83-degree side
    x_fixed_83 = -0.0507


    # ======================================================
    # Numerical derivative
    # ======================================================

    h = 5e-4


    # ======================================================
    # Cylinder integration points
    # ======================================================

    Nr = 8
    Ntheta = 20
    Nz = 4

    y_off, z_off, dV = (
        cylinder_quadrature_points(
            r_cyl,
            L_cyl,
            Nr=Nr,
            Ntheta=Ntheta,
            Nz=Nz
        )
    )


    # ======================================================
    # Force from ONE magnet row onto ONE cylinder
    # ======================================================

    def Fy_cylinder_fast(
        x_cyl,
        y_cyl,
        z_cyl,
        step_x,
        step_y,
        k_steel=1.0
    ):

        y_global = (
            y_cyl +
            y_off
        )

        z_global = (
            z_cyl +
            z_off
        )

        B_plus = Bz_total_array_vec(
            x_cyl,
            y_global + h,
            z_global,
            a,
            b,
            c,
            Mmag,
            num_magnets,
            step_x,
            step_y
        )

        B_minus = Bz_total_array_vec(
            x_cyl,
            y_global - h,
            z_global,
            a,
            b,
            c,
            Mmag,
            num_magnets,
            step_x,
            step_y
        )

        dBz_dy = (
            k_steel *
            (
                B_plus -
                B_minus
            )
            /
            (2 * h)
        )

        Fy = np.sum(
            M_cyl *
            dBz_dy *
            dV
        )

        return Fy


    # ======================================================
    # Physical parameters
    # ======================================================

    k_steel = 1.14

    plate_angle_77 = 77.0
    plate_angle_83 = 83.0

    # keep friction OFF first
    F_resistance = 0.0


    # ==========================================================
    # NUMBER OF MAGNETS SWEEP
    # ==========================================================

    # Keep all other parameters constant
    t_cm = 1.2
    t = t_cm / 100.0

    # Cube width = 1 cm
    # Centre-to-centre spacing
    step_y = 0.01 + t

    # Fixed asymmetric plate geometry
    plate_angle_77 = 77.0
    plate_angle_83 = 83.0

    # Measured lateral positions
    x_fixed_77 = +0.056
    x_fixed_83 = -0.0507

    # Steel correction
    k_steel = 1.14

    # Provisional constant effective resistance
    F_resistance = 0.03  # N = 30 mN


    # ==========================================================
    # Geometry of the two sides
    # ==========================================================

    step_x_77 = (
        np.tan(
            np.pi / 2 -
            np.deg2rad(plate_angle_77)
        )
        * step_y
    )

    step_x_83 = (
        np.tan(
            np.pi / 2 -
            np.deg2rad(plate_angle_83)
        )
        * step_y
    )


    # ==========================================================
    # Sweep number of magnets from 1 to 8
    # ==========================================================

    magnet_numbers = np.arange(1, 9)

    avg_accels_raw = []
    avg_accels_corrected = []


    for N in magnet_numbers:

        # IMPORTANT:
        # this is the ONLY swept parameter
        num_magnets = N

        ay_raw = np.zeros_like(y_vals)
        ay_corrected = np.zeros_like(y_vals)

        for i, y0 in enumerate(y_vals):

            # ------------------------------------------
            # 77° side
            # ------------------------------------------

            Fy_77 = Fy_cylinder_fast(
                x_fixed_77,
                y0,
                z0,
                step_x_77,
                step_y,
                k_steel=k_steel
            )

            # ------------------------------------------
            # 83° side
            # ------------------------------------------

            Fy_83 = Fy_cylinder_fast(
                x_fixed_83,
                y0,
                z0,
                step_x_83,
                step_y,
                k_steel=k_steel
            )

            # Total magnetic force
            Fy_magnetic = Fy_77 + Fy_83

            # ------------------------------------------
            # RAW THEORY
            # ------------------------------------------

            ay_raw[i] = (
                Fy_magnetic /
                M_effective
            )

            # ------------------------------------------
            # THEORY WITH 30 mN RESISTANCE
            # ------------------------------------------

            Fy_net = (
                Fy_magnetic -
                F_resistance
            )

            ay_corrected[i] = (
                Fy_net /
                M_effective
            )


        # ======================================================
        # Average over same trajectory
        # ======================================================

        a_raw = (
            np.trapezoid(
                ay_raw,
                y_vals
            )
            /
            (y_vals[-1] - y_vals[0])
        )

        a_corrected = (
            np.trapezoid(
                ay_corrected,
                y_vals
            )
            /
            (y_vals[-1] - y_vals[0])
        )


        avg_accels_raw.append(a_raw)
        avg_accels_corrected.append(a_corrected)


        print(
            f"N = {N} magnets "
            f"| Raw = {a_raw:.4f} m/s² "
            f"| With 30 mN resistance = "
            f"{a_corrected:.4f} m/s²"
        )


    # ==========================================================
    # Convert to arrays
    # ==========================================================

    avg_accels_raw = np.array(avg_accels_raw)
    avg_accels_corrected = np.array(avg_accels_corrected)


    # ==========================================================
    # Plot
    # ==========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        magnet_numbers,
        avg_accels_raw,
        marker="o",
        label="Magnetic theory"
    )

    plt.plot(
        magnet_numbers,
        avg_accels_corrected,
        marker="o",
        linestyle="--",
        label="Theory with 30 mN effective resistance"
    )

    plt.xlabel("Number of cube magnets per side")
    plt.ylabel("Average acceleration (m/s²)")

    plt.title(
        "Average Acceleration vs Number of Magnets\n"
        "(t = 1.2 cm, 77° / 83° geometry)"
    )

    plt.xticks(magnet_numbers)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    # ==========================================================
    # EXPERIMENTAL DATA
    # ==========================================================

    experimental_N = np.array([
        1, 2, 3, 4, 5, 6, 7, 8
    ])

    experimental_a = np.array([
        0.037808556,
        0.128455951,
        0.24030491,
        0.270992627,
        0.250021209,
        0.351838259,
        0.402998,
        0.3940776
    ])

    experimental_err = np.array([
        0.022652615,
        0.048540607,
        0.133972652,
        0.140193251,
        0.096880147,
        0.111238021,
        0.038761895,
        0.017363004
    ])


    # ==========================================================
    # PLOT THEORY + EXPERIMENT
    # ==========================================================

    plt.figure(figsize=(10, 6))

    # raw magnetic theory
    plt.plot(
        magnet_numbers,
        avg_accels_raw,
        marker="o",
        label="Theory without resistance"
    )

    # theory including 30 mN effective resistance
    plt.plot(
        magnet_numbers,
        avg_accels_corrected,
        marker="o",
        linestyle="--",
        label="Theory with 30 mN resistance"
    )

    # experimental results with standard errors
    plt.errorbar(
        experimental_N,
        experimental_a,
        yerr=experimental_err,
        fmt="o",
        capsize=5,
        label="Experiment"
    )

    plt.xlabel("Number of cube magnets per side")
    plt.ylabel("Average acceleration (m/s²)")

    plt.title(
        "Acceleration vs Number of Magnets\n"
        "(t = 1.2 cm, 77° / 83° geometry)"
    )

    plt.xticks(experimental_N)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ==========================================================
    # NUMERICAL COMPARISON
    # ==========================================================

    print("\nTHEORY VS EXPERIMENT")
    print("------------------------------------------------------------")

    for N, exp, err, raw, corrected in zip(
        experimental_N,
        experimental_a,
        experimental_err,
        avg_accels_raw,
        avg_accels_corrected
    ):

        residual_raw = raw - exp
        residual_corrected = corrected - exp

        sigma_raw = residual_raw / err
        sigma_corrected = residual_corrected / err

        print(
            f"N = {N} | "
            f"Exp = {exp:.4f} ± {err:.4f} | "
            f"Raw = {raw:.4f} ({sigma_raw:+.2f}σ) | "
            f"30mN = {corrected:.4f} ({sigma_corrected:+.2f}σ)"
        )

    # ==========================================================
    # RMSE: theory versus experiment
    # ==========================================================

    # Theory without resistance.
    rmse_raw = np.sqrt(
        np.mean(
            (avg_accels_raw - experimental_a) ** 2
        )
    )

    # Theory with 30 mN resistance.
    rmse_corrected = np.sqrt(
        np.mean(
            (avg_accels_corrected - experimental_a) ** 2
        )
    )

    print(
        f"\nRMSE without resistance: "
        f"{rmse_raw:.4f} m/s²"
    )

    print(
        f"RMSE with "
        f"{F_resistance * 1000:.0f} mN resistance: "
        f"{rmse_corrected:.4f} m/s²"
    )