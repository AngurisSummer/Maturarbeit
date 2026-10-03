import numpy as np
import matplotlib.pyplot as plt


# ==========================================================
# Physical constant
# ==========================================================

mu0 = 4 * np.pi * 1e-7


# ==========================================================
# Magnetic field of one rectangular cube magnet
# ==========================================================

def F1_fn(x, y, z, a, b, c):

    R = np.sqrt(
        (x + a) ** 2
        + (y + b) ** 2
        + (z + c) ** 2
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
# Integration points inside the cylindrical magnet
# ==========================================================

def cylinder_quadrature_points(
    r_cyl,
    L_cyl,
    Nr=8,
    Ntheta=20,
    Nz=4
):

    rho_edges = np.linspace(
        0.0,
        r_cyl,
        Nr + 1
    )

    rho = 0.5 * (
        rho_edges[:-1]
        + rho_edges[1:]
    )

    dr = (
        rho_edges[1:]
        - rho_edges[:-1]
    )

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
        z_edges[:-1]
        + z_edges[1:]
    )

    dz = (
        z_edges[1:]
        - z_edges[:-1]
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
        RHO
        * DR
        * DTH
        * DZ
    ).ravel()

    return y_off, z_off, dV


# ==========================================================
# Combined magnetic field from one row of cube magnets
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

    a = 0.005

    b = 0.005

    c = 0.005

    Mmag = 1.05e6

    num_magnets = 4

    # Cube width: 10 mm
    # Cube-to-cube gap: 12 mm
    # Center-to-center spacing: 22 mm

    step_y = 0.022


    # ======================================================
    # Moving cylindrical magnets
    # ======================================================

    r_cyl = 0.0175

    L_cyl = 0.004

    Br_cyl = 1.21

    M_cyl = Br_cyl / mu0


    # ======================================================
    # Effective inertial mass
    # ======================================================

    M_physical = 0.07125

    m_disc = 0.027

    r_wheel = 0.0175

    # Two solid cylindrical magnetic wheels

    M_effective = (
        M_physical
        + m_disc
    )

    print(
        f"Effective mass = "
        f"{M_effective:.6f} kg"
    )


    # ======================================================
    # Geometry
    # ======================================================

    z0 = 0.0175

    y_vals = np.linspace(
        0.025,
        0.22,
        250
    )

    # Variable-angle row on the positive-x side

    x_variable_row = +0.056

    # Fixed-angle row on the negative-x side

    x_fixed_row = -0.0507

    fixed_row_angle_deg = 83.0


    # ======================================================
    # Numerical settings
    # ======================================================

    h = 5e-4

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
    # Force from one magnet row on one cylindrical magnet
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
            y_cyl + y_off
        )

        z_global = (
            z_cyl + z_off
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
            k_steel
            * (B_plus - B_minus)
            / (2 * h)
        )

        Fy = np.sum(
            M_cyl
            * dBz_dy
            * dV
        )

        return Fy


    # ======================================================
    # Physical corrections
    # ======================================================

    k_steel = 1.14

    # Adjust this value if you have an independently
    # measured resistance force.

    F_resistance = 0.05


    # ======================================================
    # Experimental angle-sweep data
    # ======================================================

    experimental_angles = np.array([
        75.0,
        77.0,
        80.0,
        83.0
    ])

    experimental_a = np.array([
        0.2245326,
        0.1868504,
        0.1486434,
        0.1128628
    ])

    experimental_err = np.array([
        0.0352,
        0.0201,
        0.0041,
        0.0178
    ])

    # Angle uncertainty in degrees.
    # Change this to match the actual precision
    # of your angle measurement.

    experimental_angle_err = 1.0


    # ======================================================
    # Theoretical angle sweep
    # ======================================================

    angles_deg = np.linspace(
        75.0,
        83.0,
        33
    )

    raw_theory = []

    corrected_theory = []

    variable_row_contributions = []

    fixed_row_contributions = []


    # ======================================================
    # Geometry of the fixed 83° row
    # ======================================================

    # Negative sign because this row lies on the
    # negative-x side and tapers toward the cylinder.

    step_x_fixed = -(
        np.tan(
            np.deg2rad(
                90.0 - fixed_row_angle_deg
            )
        )
        * step_y
    )


    # ======================================================
    # Calculate fixed-row force once
    # ======================================================

    fixed_row_force_profile = np.zeros_like(
        y_vals
    )

    for i, y0 in enumerate(y_vals):

        fixed_row_force_profile[i] = (
            Fy_cylinder_fast(
                x_fixed_row,
                y0,
                z0,
                step_x_fixed,
                step_y,
                k_steel=k_steel
            )
        )


    # ======================================================
    # Sweep the angle of the other row
    # ======================================================

    for variable_angle_deg in angles_deg:

        # Positive sign because this row lies on the
        # positive-x side and tapers toward the cylinder.

        step_x_variable = (
            np.tan(
                np.deg2rad(
                    90.0 - variable_angle_deg
                )
            )
            * step_y
        )

        variable_row_force_profile = np.zeros_like(
            y_vals
        )

        raw_acceleration_profile = np.zeros_like(
            y_vals
        )

        corrected_acceleration_profile = np.zeros_like(
            y_vals
        )

        for i, y0 in enumerate(y_vals):

            # Force from the variable-angle row

            Fy_variable = Fy_cylinder_fast(
                x_variable_row,
                y0,
                z0,
                step_x_variable,
                step_y,
                k_steel=k_steel
            )

            variable_row_force_profile[i] = (
                Fy_variable
            )

            # Force from the fixed 83° row

            Fy_fixed = (
                fixed_row_force_profile[i]
            )

            # Explicit combined magnetic force

            Fy_magnetic = (
                Fy_variable
                + Fy_fixed
            )

            # Theory without resistance

            raw_acceleration_profile[i] = (
                Fy_magnetic
                / M_effective
            )

            # Theory including resistance

            corrected_acceleration_profile[i] = (
                (
                    Fy_magnetic
                    - F_resistance
                )
                / M_effective
            )


        # ==================================================
        # Spatially averaged accelerations
        # ==================================================

        interval_length = (
            y_vals[-1]
            - y_vals[0]
        )

        a_raw = (
            np.trapezoid(
                raw_acceleration_profile,
                y_vals
            )
            / interval_length
        )

        a_corrected = (
            np.trapezoid(
                corrected_acceleration_profile,
                y_vals
            )
            / interval_length
        )

        a_variable_row = (
            np.trapezoid(
                variable_row_force_profile,
                y_vals
            )
            / interval_length
            / M_effective
        )

        a_fixed_row = (
            np.trapezoid(
                fixed_row_force_profile,
                y_vals
            )
            / interval_length
            / M_effective
        )

        raw_theory.append(
            a_raw
        )

        corrected_theory.append(
            a_corrected
        )

        variable_row_contributions.append(
            a_variable_row
        )

        fixed_row_contributions.append(
            a_fixed_row
        )

        print(
            f"Angle = {variable_angle_deg:.2f}° | "
            f"Variable row = {a_variable_row:.4f} m/s² | "
            f"Fixed row = {a_fixed_row:.4f} m/s² | "
            f"Raw theory = {a_raw:.4f} m/s² | "
            f"Corrected theory = {a_corrected:.4f} m/s²"
        )


    # ======================================================
    # Convert theoretical results to NumPy arrays
    # ======================================================

    raw_theory = np.array(
        raw_theory
    )

    corrected_theory = np.array(
        corrected_theory
    )


    # ======================================================
    # RMSE at experimental angles
    # ======================================================

    raw_at_experimental_angles = np.interp(
        experimental_angles,
        angles_deg,
        raw_theory
    )

    corrected_at_experimental_angles = np.interp(
        experimental_angles,
        angles_deg,
        corrected_theory
    )

    rmse_raw = np.sqrt(
        np.mean(
            (
                raw_at_experimental_angles
                - experimental_a
            ) ** 2
        )
    )

    rmse_corrected = np.sqrt(
        np.mean(
            (
                corrected_at_experimental_angles
                - experimental_a
            ) ** 2
        )
    )

    print(
        f"\nRMSE without resistance: "
        f"{rmse_raw:.4f} m/s²"
    )

    print(
        f"RMSE with resistance: "
        f"{rmse_corrected:.4f} m/s²"
    )


    # ======================================================
    # Plot both theoretical curves and experimental data
    # ======================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        angles_deg,
        raw_theory,
        color="tab:blue",
        linewidth=2,
        label="Theory without resistance"
    )

    plt.plot(
        angles_deg,
        corrected_theory,
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label=(
            f"Theory with "
            f"{F_resistance * 1000:.0f} mN resistance"
        )
    )

    plt.errorbar(
        experimental_angles,
        experimental_a,
        xerr=experimental_angle_err,
        yerr=experimental_err,
        fmt="o",
        color="tab:green",
        ecolor="tab:green",
        capsize=5,
        markersize=7,
        label="Experiment"
    )

    plt.xlabel(
        "Variable magnet-row angle (degrees)"
    )

    plt.ylabel(
        "Average acceleration (m/s²)"
    )

    plt.title(
        "Average Acceleration vs Angle of (normally) 77° side\n"
        "(t = 1.2 cm, 83° geometry)"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()