import numpy as np
import matplotlib.pyplot as plt

# Debug mode
debug = False

import numpy as np



def xgit_prop_curve(t, debug=True):
    # Define the curve before soft grooming
    expConst = -0.2

    if t < 11:
        value = -17500 * np.exp(expConst*t)
        if debug:
            print(f"xgit_prop_curve returns: {value} for t={t} (before soft grooming)")
        return value
    # Define the drop due to soft grooming
    elif t >= 10:
        value = (-10000 + (7500 * np.exp(expConst*(t-8))))
        if debug:
            print(f"xgit_prop_curve returns: {value} for t={t} (after soft grooming)")
        return value

def polishing_curve(t, debug=True):
    # Define the curve before first polishing
    if t < 6:
        value = -1500 * t
        if debug:
            print(f"polishing_curve returns: {value} for t={t} (before first polishing)")
        return value
    # Define the drop due to first polishing
    elif t == 6:
        value = -9000
        if debug:
            print(f"polishing_curve returns: {value} for t={t} (first polishing)")
        return value
    # Define the curve between two polishings
    elif t < 12:
        value = (-9000 - 1500 * (t-6))
        if debug:
            print(f"polishing_curve returns: {value} for t={t} (between two polishings)")
        return value
    # Define the drop due to second polishing
    elif t == 12:
        value = -18000
        if debug:
            print(f"polishing_curve returns: {value} for t={t} (second polishing)")
        return value
    # Define the curve after second polishing
    else:
        value = (-18000 - 1500 * (t-12))
        if debug:
            print(f"polishing_curve returns: {value} for t={t} (after second polishing)")
        return value


# New function to calculate the total ROI for XGIT-PROP over a given period
def total_roi_xgit_prop(t_end):
    t_values = np.arange(0, t_end, 1)
    roi_values = [xgit_prop_curve(ti) for ti in t_values]
    return np.trapz(roi_values, t_values)

# New function to calculate the total ROI for Polishing over a given period
def total_roi_polishing(t_end):
    t_values = np.arange(0, t_end, 1)
    roi_values = [polishing_curve(ti) for ti in t_values]
    return np.trapz(roi_values, t_values)

# Generate the data
#generate a list of integer values starting at 0 and ending at 16 using arange
t = np.arange(0, 16, 1)
xgit_prop_values = [xgit_prop_curve(ti) for ti in t]
polishing_values = [polishing_curve(ti) for ti in t]

data = plt.read_csv("prop_polishing_data_updated.csv")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data["timestamp"], xgit_prop_values, label="XGIT-PROP", color="green")
plt.plot(data["timestamp"], polishing_values, label="Polishing", color="blue")

# Proactive cleaning zone
plt.axvspan(8, 12, alpha=0.2, color='grey')

# Annotations and titles
plt.axhline(y=0, color='black',linewidth=0.5)
plt.axhline(-17,500, color='black',linewidth=0.5, linestyle="--")
plt.title("XGIT PROP vs Polishing")
plt.xlabel("Time (months)")
plt.ylabel("ROI ($)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

# Display the plot
plt.show()

