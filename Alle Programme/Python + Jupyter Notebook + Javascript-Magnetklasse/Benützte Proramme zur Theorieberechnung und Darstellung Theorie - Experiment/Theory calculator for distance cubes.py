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


    # ======================================================
    # CUBE GAP SWEEP
    # ======================================================

    t_values_cm = [
        0.6,
        1.1,
        1.2,
        1.6,
        2.1,
        2.6,
        3.1,
        3.6
    ]

    avg_accels = []


    for t_cm in t_values_cm:

        # ----------------------------------------------
        # Centre-to-centre y spacing
        # ----------------------------------------------

        t = t_cm / 100.0

        step_y = (
            0.01 +
            t
        )


        # ----------------------------------------------
        # 77-degree side geometry
        # ----------------------------------------------

        step_x_77 = (
            np.tan(
                np.pi / 2
                -
                np.deg2rad(
                    plate_angle_77
                )
            )
            *
            step_y
        )


        # ----------------------------------------------
        # 83-degree side geometry
        # ----------------------------------------------

        step_x_83 = (
            np.tan(
                np.pi / 2
                -
                np.deg2rad(
                    plate_angle_83
                )
            )
            *
            step_y
        )


        ay_vals = np.zeros_like(
            y_vals
        )


        for i, y0 in enumerate(
            y_vals
        ):

            # ------------------------------------------
            # Force from 77-degree side
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
            # Force from 83-degree side
            # ------------------------------------------

            Fy_83 = Fy_cylinder_fast(
                x_fixed_83,
                y0,
                z0,
                step_x_83,
                step_y,
                k_steel=k_steel
            )


            # ------------------------------------------
            # Total magnetic force
            # ------------------------------------------

            Fy_total = (
                Fy_77 +
                Fy_83
            )


            # optional rolling resistance
            Fy_net = (
                Fy_total -
                F_resistance
            )


            # ------------------------------------------
            # Vehicle acceleration
            # ------------------------------------------

            ay_vals[i] = (
                Fy_net /
                M_effective
            )


        # ==================================================
        # Spatial average
        # ==================================================

        a_avg = (
            np.trapezoid(
                ay_vals,
                y_vals
            )
            /
            (
                y_vals[-1] -
                y_vals[0]
            )
        )

        avg_accels.append(
            a_avg
        )


        print(
            f"t = {t_cm:.2f} cm "
            f"-> Avg acceleration = "
            f"{a_avg:.4f} m/s²"
        )


    # ======================================================
    # Experimental data
    # ======================================================

    experimental_t = np.array([
        0.6,
        1.1,
        1.6,
        2.1,
        2.6,
        3.1,
        3.6
    ])

    experimental_a = np.array([
        0.3463595,
        0.3384158,
        0.2969192,
        0.1749574,
        0.1515178,
        0.1187936,
        0.1092068
    ])

    experimental_err = np.array([
        0.0348055,
        0.01787546,
        0.02957192,
        0.01762536,
        0.01574086,
        0.02326507,
        0.03868291
    ])


    # ======================================================
    # Plot
    # ======================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        t_values_cm,
        avg_accels,
        marker="o",
        label="Theory"
    )

    plt.errorbar(
        experimental_t,
        experimental_a,
        xerr=0.1,  
        yerr=experimental_err,
        fmt="o",
        capsize=5,
        label="Experiment"
    )

    plt.xlabel(
        "Cube-to-cube gap t (cm)"
    )

    plt.ylabel(
        "Average acceleration (m/s²)"
    )

    plt.title(
        "Distance Sweep: "
        "Explicit 77° + 83° Sides"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    # ======================================================
    # Plot theory shifted down by 0.3 m/s²
    # ======================================================

    theory_shift = 0.3

    shifted_avg_accels = (
        np.array(avg_accels) - theory_shift
    )

    plt.figure(figsize=(10, 6))

    # shifted theory
    plt.plot(
        t_values_cm,
        shifted_avg_accels,
        marker="o",
        label="Theory shifted by -0.20 m/s²"
    )

    # experimental data
    plt.errorbar(
        experimental_t,
        experimental_a,
        xerr=0.1,  
        yerr=experimental_err,
        fmt="o",
        capsize=5,
        label="Experiment"
    )

    plt.xlabel("Cube-to-cube gap t (cm)")
    plt.ylabel("Average acceleration (m/s²)")

    plt.title(
        "Distance Sweep: Theory shifted by -0.20 m/s²"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ======================================================
    # Print shifted values
    # ======================================================

    print("\nShifted theoretical values:")

    for t_cm, original, shifted in zip(
        t_values_cm,
        avg_accels,
        shifted_avg_accels
    ):
        print(
            f"t = {t_cm:.2f} cm "
            f"| original = {original:.4f} "
            f"| shifted = {shifted:.4f} m/s²"
        )

    # ======================================================
    # Regular theory and shifted theory on the same graph
    # ======================================================

    theory_shift = 0.3

    shifted_avg_accels = (
        np.array(avg_accels) - theory_shift
    )

    plt.figure(figsize=(10, 6))

    # Regular theory
    plt.plot(
        t_values_cm,
        avg_accels,
        marker="o",
        label="Regular theory"
    )

    # Shifted theory
    plt.plot(
        t_values_cm,
        shifted_avg_accels,
        marker="o",
        linestyle="--",
        label="Theory shifted by 21 mN"
    )

    # Experimental data
    plt.errorbar(
        experimental_t,
        experimental_a,
        xerr=0.1,                 # ±0.1 cm = ±1 mm
        yerr=experimental_err,
        fmt="o",
        capsize=5,
        label="Experiment"
    )

    plt.xlabel("Cube-to-cube gap t (cm)")
    plt.ylabel("Average acceleration (m/s²)")
    plt.title(
        "Average Acceleration vs Distance between magnets in direction of acceleration\n"
        "(77° / 83° geometry)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ======================================================
    # RMSE: theory versus experiment
    # ======================================================

    # Match theoretical predictions to experimental x-values.
    # This is necessary because theory includes an extra point at 1.2 cm.
    theory_at_experiment = np.interp(
        experimental_t,
        t_values_cm,
        avg_accels
    )

    shifted_theory_at_experiment = np.interp(
        experimental_t,
        t_values_cm,
        shifted_avg_accels
    )

    # Calculate RMSE for regular theory.
    rmse_regular = np.sqrt(
        np.mean(
            (theory_at_experiment - experimental_a) ** 2
        )
    )

    # Calculate RMSE for shifted theory.
    rmse_shifted = np.sqrt(
        np.mean(
            (shifted_theory_at_experiment - experimental_a) ** 2
        )
    )

    print(f"RMSE regular theory: {rmse_regular:.4f} m/s²")
    print(f"RMSE shifted theory: {rmse_shifted:.4f} m/s²")