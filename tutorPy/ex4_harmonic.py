import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the harmonic motion
amplitude = 2.0
frequency = 0.5  # Hz
phase = np.pi/4  # 45 degrees phase offset
damping = 0.1    # Damping coefficient

# Create time array from 0 to 10 seconds with 1000 points
t = np.linspace(0, 10, 1000)

# Calculate the position over time (damped harmonic oscillator)
position = amplitude * np.exp(-damping * t) * np.sin(2 * np.pi * frequency * t + phase)

# Calculate velocity (derivative of position)
velocity = amplitude * np.exp(-damping * t) * (
    2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t + phase) - 
    damping * np.sin(2 * np.pi * frequency * t + phase)
)

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot position vs time
ax1.plot(t, position, label=f'Position (A={amplitude}, f={frequency}Hz)')
ax1.set_ylabel('Position (m)')
ax1.set_title('Harmonic Movement - Position vs Time')
ax1.grid(True)
ax1.legend()

# Plot velocity vs time
ax2.plot(t, velocity, 'r', label='Velocity')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('Velocity vs Time')
ax2.grid(True)
ax2.legend()

plt.tight_layout()

# Animation of the harmonic motion
fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
ax_anim.set_xlim(0, 10)
ax_anim.set_ylim(-amplitude*1.1, amplitude*1.1)
ax_anim.set_xlabel('Time (s)')
ax_anim.set_ylabel('Position (m)')
ax_anim.set_title('Harmonic Motion Animation')
ax_anim.grid(True)

line, = ax_anim.plot([], [], 'b-', lw=2)
point, = ax_anim.plot([], [], 'ro', ms=10)
time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)

def init():
    line.set_data([], [])
    point.set_data([], [])
    time_text.set_text('')
    return line, point, time_text

def animate(i):
    # Only plot up to current frame
    current_t = t[:i]
    current_pos = position[:i]
    line.set_data(current_t, current_pos)
    point.set_data(current_t[-1], current_pos[-1])
    time_text.set_text(f'Time = {current_t[-1]:.2f}s')
    return line, point, time_text

ani = FuncAnimation(fig_anim, animate, frames=len(t), init_func=init,
                    blit=True, interval=20, repeat=True)

plt.tight_layout()
plt.show()