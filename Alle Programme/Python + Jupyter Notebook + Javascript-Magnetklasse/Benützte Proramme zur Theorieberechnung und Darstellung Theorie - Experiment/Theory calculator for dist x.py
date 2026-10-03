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

    # two solid cylindrical magnetic rollers
    m_disc = 0.027
    r_wheel = 0.0175

    M_effective = (
        M_physical
        +
        m_disc
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
    # EDGE-DISTANCE SWEEP
    # ONLY THE 83° ROW MOVES
    # ==========================================================

    # Experimental positions
    edge_distances_mm = np.array([
        5, 7, 9, 11, 13, 15, 17
    ])

    experimental_a = np.array([
        0.452467,
        0.4143942,
        0.50117267,
        0.41778733,
        0.422683,
        0.371518,
        0.94796775
    ])

    experimental_err = np.array([
        0.03411016,
        0.03949804,
        0.06351325,
        0.06629884,
        0.1027859,
        0.11973754,
        0.08273723
    ])


    # ==========================================================
    # Fixed parameters
    # ==========================================================

    t_cm = 1.2
    t = t_cm / 100.0

    step_y = 0.01 + t

    plate_angle_77 = 77.0
    plate_angle_83 = 83.0

    step_x_77 = (
        np.tan(
            np.pi / 2 -
            np.deg2rad(plate_angle_77)
        )
        * step_y
    )

    # opposite taper direction for second side
    step_x_83 = -(
        np.tan(
            np.pi / 2 -
            np.deg2rad(plate_angle_83)
        )
        * step_y
    )


    # 77° side stays fixed
    x_fixed_77 = +0.056


    # ==========================================================
    # Reference position
    # ==========================================================

    # IMPORTANT:
    # set this to whichever experimental edge distance
    # corresponds to x_fixed_83 = -0.0507 m
    reference_edge_mm = 5.0

    x83_reference = -0.0507


    # provisional effective resistance
    F_resistance = 0.020   # 20 mN


    raw_theory = []
    corrected_theory = []


    # ==========================================================
    # Sweep
    # ==========================================================

    for edge_mm in edge_distances_mm:

        # change relative to reference
        delta_x = (
            edge_mm - reference_edge_mm
        ) / 1000.0

        x_fixed_83 = (
            x83_reference -
            delta_x
        )

        ay_raw = np.zeros_like(y_vals)
        ay_corrected = np.zeros_like(y_vals)

        for i, y0 in enumerate(y_vals):

            # ------------------------------------------
            # 77° side — fixed
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
            # 83° side — moving inward
            # ------------------------------------------

            Fy_83 = Fy_cylinder_fast(
                x_fixed_83,
                y0,
                z0,
                step_x_83,
                step_y,
                k_steel=k_steel
            )


            Fy_magnetic = (
                Fy_77 +
                Fy_83
            )


            # raw magnetic + rotational theory
            ay_raw[i] = (
                Fy_magnetic /
                M_effective
            )


            # theory with provisional 10 mN resistance
            Fy_net = (
                Fy_magnetic -
                F_resistance
            )

            ay_corrected[i] = (
                Fy_net /
                M_effective
            )


        # ======================================================
        # Average acceleration
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

        raw_theory.append(a_raw)
        corrected_theory.append(a_corrected)

        print(
            f"Edge distance = {edge_mm:.0f} mm | "
            f"x83 = {x_fixed_83*1000:.1f} mm | "
            f"Raw = {a_raw:.4f} m/s² | "
            f"30 mN = {a_corrected:.4f} m/s²"
        )


    # ==========================================================
    # Plot
    # ==========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        edge_distances_mm,
        raw_theory,
        marker="o",
        label="Theory without resistance"
    )

    plt.plot(
        edge_distances_mm,
        corrected_theory,
        marker="o",
        linestyle="--",
        label="Theory with 20 mN resistance"
    )

    plt.errorbar(
        edge_distances_mm,
        experimental_a,
        xerr=1.0,              # ±1 mm
        yerr=experimental_err,
        fmt="o",
        capsize=5,
        label="Experiment"
    )
    plt.xlabel(
        "83° magnet-row edge to plate-edge distance (mm)"
    )

    plt.ylabel(
        "Average acceleration (m/s²)"
    )

    plt.title(
        "Average Acceleration vs Position of Magnets on 83° side\n"
        "(t = 1.2 cm, 77° / 83° geometry)"
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ======================================================
    # RMSE: theory versus experiment
    # ======================================================

    # Convert theoretical results from lists to NumPy arrays.
    raw_theory_array = np.array(raw_theory)

    corrected_theory_array = np.array(corrected_theory)

    # RMSE for theory without resistance.
    rmse_raw = np.sqrt(
        np.mean(
            (raw_theory_array - experimental_a) ** 2
        )
    )

    # RMSE for theory with resistance.
    rmse_corrected = np.sqrt(
        np.mean(
            (corrected_theory_array - experimental_a) ** 2
        )
    )

    print(
        f"RMSE theory without resistance: "
        f"{rmse_raw:.4f} m/s²"
    )

    print(
        f"RMSE theory with "
        f"{F_resistance * 1000:.0f} mN resistance: "
        f"{rmse_corrected:.4f} m/s²"
    )