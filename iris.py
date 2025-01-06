import matplotlib.pyplot as plt
import numpy as np
# Room and receiver positions
receivers = {
   "R1": (-2, 0),
   "R2": (0, 4),
   "R3": (2, 0)
}
source = (0, -4)
# Colors and time intervals
time_intervals = [(0, 2, "red"), (2, 20, "orange"), (20, 80, "green"), (80, 200, "blue"), (200, 400, "lightblue")]
# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
# Plot room layout
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.plot([-5, 5, 5, -5, -5], [-5, -5, 5, 5, -5], 'k-')  # Rectangular room boundary
# Plot source position
ax.scatter(*source, color='black', s=100, label="S")
# Plot receiver positions
for label, pos in receivers.items():
   ax.scatter(*pos, color='black', s=100)
   ax.text(pos[0] + 0.2, pos[1], label, fontsize=12, color="black")
# Add IRIS plot vectors
for label, pos in receivers.items():
   for start, end, color in time_intervals:
       # Generate random vectors for visualization
       angles = np.linspace(0, 2 * np.pi, 12)
       magnitudes = np.random.uniform(0.5, 1.5, len(angles))  # Adjust this range for magnitude randomness
       for angle, mag in zip(angles, magnitudes):
           dx, dy = mag * np.cos(angle), mag * np.sin(angle)
           ax.arrow(pos[0], pos[1], dx, dy, head_width=0.1, head_length=0.1, color=color, alpha=0.7)
# Add legend
legend_labels = [f"{start}-{end} ms" for start, end, _ in time_intervals]
legend_colors = [color for _, _, color in time_intervals]
for label, color in zip(legend_labels, legend_colors):
   ax.plot([], [], color=color, label=label, linewidth=6)
ax.legend(title="Time (ms)", loc="upper left")
# Add title and labels
ax.set_title("IRIS Plots for Receiver Positions", fontsize=14)
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
plt.grid(False)
plt.show()