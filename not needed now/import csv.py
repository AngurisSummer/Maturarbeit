import csv
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================

CSV_FILE = "CA1_R1.csv"

NUM_CHANNELS = 10
SPACING = 0.014        # metres between sensors
SAMPLE_PERIOD = 0.001  # seconds/sample — CHANGE THIS

K_SIGMA = 4.0
MIN_RUN = 3


# ============================================================
# READ CSV
# ============================================================

samples = []
values = []

with open(CSV_FILE, "r") as csvfile:

    reader = csv.DictReader(csvfile)

    for row in reader:

        try:
            sample = float(row["Sample"])

            nums = [
                float(row[f"M{i}"])
                for i in range(NUM_CHANNELS)
            ]

        except (ValueError, KeyError):
            continue

        samples.append(sample)
        values.append(nums)


samples = np.array(samples, dtype=float)

timestamps = (
    samples - samples[0]
) * SAMPLE_PERIOD

channels = np.array(
    values,
    dtype=float
).T

print(f"Loaded {len(timestamps)} samples.")


# ============================================================
# RAW SENSOR PLOT
# ============================================================

plt.figure(figsize=(12, 6))

for ch in range(NUM_CHANNELS):

    plt.plot(
        timestamps,
        channels[ch],
        label=f"M{ch}",
        alpha=0.7
    )

plt.xlabel("Time (s)")
plt.ylabel("Sensor value")
plt.title("Raw IR sensor signals")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# FIND LONGEST EVENT
# ============================================================

def longest_true_run(mask):

    if not np.any(mask):
        return None

    idx = np.where(mask)[0]

    splits = (
        np.where(np.diff(idx) != 1)[0] + 1
    )

    runs = np.split(idx, splits)

    run = max(runs, key=len)

    return int(run[0]), int(run[-1])


# ============================================================
# EVENT DETECTION
# ============================================================

t_enter_list = np.full(
    NUM_CHANNELS,
    np.nan
)

breadths = np.zeros(
    NUM_CHANNELS
)


print("\nPer-channel detection:")


for ch in range(NUM_CHANNELS):

    y = channels[ch]

    baseline = np.median(y)

    noise = (
        1.4826
        * np.median(
            np.abs(y - baseline)
        )
    )

    if noise < 1e-6:
        noise = np.std(y) + 1e-6


    dip_strength = (
        baseline - np.min(y)
    )

    peak_strength = (
        np.max(y) - baseline
    )


    if dip_strength >= peak_strength:

        threshold = (
            baseline
            - K_SIGMA * noise
        )

        inside = y < threshold

        mode = "DIP"

    else:

        threshold = (
            baseline
            + K_SIGMA * noise
        )

        inside = y > threshold

        mode = "PEAK"


    run = longest_true_run(inside)


    if run is None:

        print(
            f"M{ch}: no event"
        )

        continue


    i0, i1 = run


    if (i1 - i0 + 1) < MIN_RUN:

        print(
            f"M{ch}: event too short"
        )

        continue


    t_enter = timestamps[i0]

    t_exit = timestamps[i1]

    breadth = (
        t_exit - t_enter
    )


    t_enter_list[ch] = t_enter

    breadths[ch] = breadth


    print(
        f"M{ch}: "
        f"{mode}, "
        f"t_enter={t_enter:.6f} s, "
        f"breadth={breadth:.6f} s"
    )


# ============================================================
# REMOVE BREADTH OUTLIERS
# ============================================================

valid_b = breadths > 0

breadths_valid = (
    breadths[valid_b]
)


if len(breadths_valid) > 0:

    med = np.median(
        breadths_valid
    )

    mad = (
        np.median(
            np.abs(
                breadths_valid - med
            )
        )
        + 1e-9
    )

    lower = med - 3 * mad

    upper = med + 3 * mad


    clean = (
        valid_b
        & (breadths >= lower)
        & (breadths <= upper)
    )

else:

    clean = np.zeros(
        NUM_CHANNELS,
        dtype=bool
    )


# ============================================================
# SENSOR POSITIONS
# ============================================================

x_positions = (
    NUM_CHANNELS
    - np.arange(NUM_CHANNELS)
) * SPACING


valid = (
    clean
    & ~np.isnan(t_enter_list)
)


t_valid = t_enter_list[valid]

x_valid = x_positions[valid]


print("\nPoints used for fit:")

for t, x in zip(t_valid, x_valid):

    print(
        f"t = {t:.6f} s, "
        f"x = {x:.4f} m"
    )


# ============================================================
# QUADRATIC FIT
#
# x(t) = A t² + B t + C
#
# comparing with:
#
# x(t) = x0 + v0 t + 1/2 a t²
#
# therefore:
#
# acceleration = 2 A
# ============================================================

if len(t_valid) >= 3:

    A, B, C = np.polyfit(
        t_valid,
        x_valid,
        2
    )


    acceleration = 2 * A


    print(
        "\n===== ACCELERATION RESULT ====="
    )

    print(
        f"x(t) = "
        f"{A:.6f} t² "
        f"+ {B:.6f} t "
        f"+ {C:.6f}"
    )

    print(
        f"Acceleration = "
        f"{acceleration:.6f} m/s²"
    )


    # ========================================================
    # FIT CURVE
    # ========================================================

    t_fit = np.linspace(
        np.min(t_valid),
        np.max(t_valid),
        300
    )


    x_fit = (
        A * t_fit**2
        + B * t_fit
        + C
    )


    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )


    plt.errorbar(
        t_valid,
        x_valid,
        yerr=0.005,
        fmt="o",
        capsize=5,
        label="Experimental points ±0.005 m"
    )


    plt.plot(
        t_fit,
        x_fit,
        linewidth=2.5,
        label="Quadratic fit"
    )


    plt.text(
        0.05,
        0.95,

        f"Acceleration = "
        f"{acceleration:.3f} m/s²",

        transform=plt.gca().transAxes,

        fontsize=14,

        verticalalignment="top",

        bbox=dict(
            facecolor="white",
            alpha=0.7
        )
    )


    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Position (m)"
    )

    plt.title(
        "Position vs Time"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


else:

    print(
        "\nNot enough valid sensors "
        "for quadratic fit."
    )