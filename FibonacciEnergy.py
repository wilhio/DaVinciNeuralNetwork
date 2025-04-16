import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

def fibonacci_sequence(n):
    """Generate Fibonacci sequence up to n terms and return mod 3 values."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])

    # Take mod 3 of each number in the sequence
    mod3_sequence = [num % 3 for num in sequence]
    return mod3_sequence

# Example usage:
n = 10  # Generate first 10 Fibonacci numbers
fib_seq_mod3 = fibonacci_sequence(n)
print(f"First {n} Fibonacci numbers mod 3: {fib_seq_mod3}")

# Process triplets
triplets = [fib_seq_mod3[i:i+3] for i in range(0, len(fib_seq_mod3), 3)]
triplet_sums = [(sum(triplet) % 3) for triplet in triplets]
print(f"Triplet sums mod 3: {triplet_sums}")

def digital_root(n):
    """Calculate the digital root of a number by summing its digits until single digit."""
    if n < 10:
        return n
    return digital_root(sum(int(d) for d in str(n)))

def fibonacci_sequence_with_digital_root(n):
    """Generate Fibonacci sequence with digital root feeding into next number."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    sequence = [0, 1]
    for i in range(2, n):
        # Get digital root of previous number
        prev_digital_root = digital_root(sequence[i-1])
        # Add it to the normal Fibonacci sum
        next_num = sequence[i-1] + sequence[i-2] + prev_digital_root
        sequence.append(next_num)

    return sequence

# Example usage:
n = 10  # Generate first 10 modified Fibonacci numbers
fib_seq_with_roots = fibonacci_sequence_with_digital_root(n)
print(f"First {n} Fibonacci numbers with digital roots: {fib_seq_with_roots}")

def fibonacci_sequence_with_digital_root_and_mod3(n):
    """Generate Fibonacci sequence with digital root feeding into next number and mod 3."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    sequence = [0, 1]
    for i in range(2, n):
        # Get digital root of previous number
        prev_digital_root = digital_root(sequence[i-1])
        # Add it to the normal Fibonacci sum
        next_num = sequence[i-1] + sequence[i-2] + prev_digital_root
        # Take mod 3 of the result
        mod3_num = next_num % 3
        sequence.append(mod3_num)

    return sequence

def fibonacci_sequence_with_digital_root_of_triplets(n):
    """Generate Fibonacci sequence and take digital root of each group of 3 numbers."""
    if n <= 0:
        return []

    # Generate the base Fibonacci sequence
    sequence = fibonacci_sequence_with_digital_root_and_mod3(n)

    # Process triplets
    triplet_roots = []
    for i in range(0, len(sequence), 3):
        triplet = sequence[i:i+3]
        if len(triplet) == 3:  # Only process complete triplets
            # Sum the triplet and get its digital root
            triplet_sum = sum(triplet)
            triplet_root = digital_root(triplet_sum)
            triplet_roots.append(triplet_root)

    return triplet_roots

# Example usage:
n = 12  # Generate enough numbers to get 4 complete triplets
triplet_roots = fibonacci_sequence_with_digital_root_of_triplets(n)
print(f"Digital roots of triplets: {triplet_roots}")

# Initialize energy reservoir and cumulative energy
energy_reservoir = 0
cumulative_energy = [1.0]  # Initialize with non-zero value to avoid division by zero

# Initialize merged_sequences with a default value to avoid empty list
merged_sequences = [1.0]  # Initialize with non-zero value

# Define strengthening factor based on energy reservoir
strengthening_factor = 1 + (energy_reservoir / max(cumulative_energy))

# Apply strengthening to simulation parameters
base_amplitude = 1.0
base_frequency = 1.0
base_phase_shift = 1.0
strengthened_params = {
    'amplitude': base_amplitude * strengthening_factor,
    'frequency': base_frequency * strengthening_factor,
    'phase_shift': base_phase_shift * strengthening_factor
}

print(f"\nStrengthening factor based on energy reservoir: {strengthening_factor:.2f}")
print("\nStrengthened simulation parameters:")
for param, value in strengthened_params.items():
    print(f"{param}: {value:.2f}")

# Visualize parameter strengthening
plt.figure(figsize=(12, 6))
plt.title('Parameter Strengthening Based on Energy Reservoir')
plt.bar(['Original', 'Strengthened'],
        [1.0, strengthening_factor],
        color=['blue', 'red'],
        alpha=0.6)
plt.ylabel('Relative Parameter Strength')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

def hilbert(x, axis=-1):
    """
    Compute the analytic signal using the Hilbert transform.

    Parameters:
    -----------
    x : array_like
        Signal data. Must be real.
    axis : int, optional
        Axis along which to do the transformation. Default is the last axis.

    Returns:
    --------
    xa : ndarray
        Analytic signal of x, with the Hilbert transform of x in the imaginary part.
    """
    N = x.shape[axis]
    X = np.fft.fft(x, axis=axis)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    if axis != -1:
        h = np.expand_dims(h, tuple(range(x.ndim-1)))

    return np.fft.ifft(X * h, axis=axis)

# Calculate energy-based metrics for advanced strengthening
# Add small epsilon to denominator to avoid division by zero
epsilon = 1e-10
energy_density = energy_reservoir / (len(merged_sequences) * max(cumulative_energy) + epsilon)
energy_stability = 1 - (np.std(cumulative_energy) / (np.mean(cumulative_energy) + epsilon))

# Define advanced strengthening factor incorporating multiple energy metrics
advanced_strengthening_factor = (
    1 +
    (energy_reservoir / (max(cumulative_energy) + epsilon)) * 0.5 +  # Base energy ratio
    energy_density * 0.3 +                                           # Energy density contribution
    energy_stability * 0.2                                          # Stability contribution
)

# Apply bounds to keep strengthening factor in reasonable range
advanced_strengthening_factor = max(1.0, min(3.0, advanced_strengthening_factor))

# Define base frequency range
frequencies = np.linspace(0.1, 10, 100)  # Frequencies from 0.1 to 10 Hz with 100 points

# Adjust frequency range based on strengthening
frequency_range = frequencies[-1] - frequencies[0]
frequencies = np.linspace(
    frequencies[0],
    frequencies[0] + (frequency_range * advanced_strengthening_factor),
    int(len(frequencies) * advanced_strengthening_factor)
)

# Calculate energy-based metrics for advanced strengthening
energy_density = energy_reservoir / (len(merged_sequences) * max(cumulative_energy))
energy_stability = 1 - (np.std(cumulative_energy) / np.mean(cumulative_energy))

# Define advanced strengthening factor incorporating multiple energy metrics
advanced_strengthening_factor = (
    1 +
    (energy_reservoir / max(cumulative_energy)) * 0.5 +  # Base energy ratio
    energy_density * 0.3 +                               # Energy density contribution
    energy_stability * 0.2                              # Stability contribution
)

# Apply bounds to keep strengthening factor in reasonable range
advanced_strengthening_factor = max(1.0, min(3.0, advanced_strengthening_factor))

# Define base frequency range
frequencies = np.linspace(0.1, 10, 100)  # Frequencies from 0.1 to 10 Hz with 100 points

# Adjust frequency range based on strengthening
frequency_range = frequencies[-1] - frequencies[0]
frequencies = np.linspace(
    frequencies[0],
    frequencies[0] + (frequency_range * advanced_strengthening_factor),
    int(len(frequencies) * advanced_strengthening_factor)
)


# Calculate energy-based metrics for advanced strengthening
energy_density = energy_reservoir / (len(merged_sequences) * max(cumulative_energy))
energy_stability = 1 - (np.std(cumulative_energy) / np.mean(cumulative_energy))

# Define advanced strengthening factor incorporating multiple energy metrics
advanced_strengthening_factor = (
    1 +
    (energy_reservoir / max(cumulative_energy)) * 0.5 +  # Base energy ratio
    energy_density * 0.3 +                               # Energy density contribution
    energy_stability * 0.2                              # Stability contribution
)

# Apply bounds to keep strengthening factor in reasonable range
advanced_strengthening_factor = max(1.0, min(3.0, advanced_strengthening_factor))

# Define base frequency range
frequencies = np.linspace(0.1, 10, 100)  # Frequencies from 0.1 to 10 Hz with 100 points


# Adjust frequency range based on strengthening
frequency_range = frequencies[-1] - frequencies[0]
frequencies = np.linspace(
    frequencies[0],
    frequencies[0] + (frequency_range * advanced_strengthening_factor),
    int(len(frequencies) * advanced_strengthening_factor)
)

!pip install scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert # Import the hilbert function

# Define simulation parameters
time_steps = 1000   # Number of time steps for wave simulation
frequencies = np.linspace(0.1, 10, 100)   # Frequency range to test
dimensionality = 1024   # High-dimensional vector space for HDC simulation

# Define advanced strengthening factor
advanced_strengthening_factor = 2.0  # Example value, adjust as needed

# Scale random projection matrix to match new dimensionality
random_projection = np.random.randn(dimensionality, len(frequencies)) * np.sqrt(advanced_strengthening_factor)

# Reinitialize wave data array with strengthened dimensions
wave_data = np.zeros((len(frequencies), time_steps))

# Apply strengthening to simulation parameters based on advanced strengthening factor
time_steps = int(time_steps * advanced_strengthening_factor)  # Increase temporal resolution
dimensionality = int(dimensionality * advanced_strengthening_factor)  # Expand vector space

# Generate waveform data in high-dimensional space
wave_data = np.zeros((len(frequencies), time_steps))

for i, freq in enumerate(frequencies):
    t = np.linspace(0, 2 * np.pi, time_steps)
    wave_data[i, :] = np.sin(freq * t)   # Simple sinusoidal wave

# Apply Hilbert Transform to get the analytic signal (used for detecting phase shifts)
analytic_signal = hilbert(wave_data, axis=1)
amplitude_envelope = np.abs(analytic_signal)   # Extract envelope

# Simulate high-dimensional encoding using random projection
random_projection = np.random.randn(dimensionality, len(frequencies))
hdc_encoded_waveforms = np.dot(random_projection, wave_data)

# Detect anomalies by checking energy conservation in the high-dimensional space
energy_levels = np.sum(hdc_encoded_waveforms ** 2, axis=1)
energy_anomalies = np.abs(np.diff(energy_levels[: len(frequencies) - 1]))   # Look for unexpected fluctuations

# Ensure correct array dimensions for plotting
energy_anomalies = energy_anomalies[: len(frequencies) - 2]

# Extract detected anomalous frequencies
anomalous_frequencies = frequencies[:-2][energy_anomalies > np.mean(energy_anomalies) + 2 * np.std(energy_anomalies)]

# Ensure correct indexing of anomalous frequencies
valid_indices = np.where(np.isin(frequencies[:-2], anomalous_frequencies))[0]

# Extract waveform data for anomalous frequencies
anomalous_waveforms = wave_data[valid_indices, :]

# Compute the phase shift using Hilbert transform for anomalies
anomalous_phase_shifts = np.angle(hilbert(anomalous_waveforms, axis=1))

# Compute the energy distribution over time for anomalies
anomalous_energy_levels = np.sum(anomalous_waveforms**2, axis=1)

# Compute phase shift derivatives to identify abrupt changes
phase_shift_derivatives = np.diff(anomalous_phase_shifts, axis=1)

# Compute second derivatives to detect sharp discontinuities
phase_shift_second_derivatives = np.diff(phase_shift_derivatives, axis=1)

# Identify locations of strongest phase discontinuities
discontinuity_threshold = np.mean(phase_shift_second_derivatives) + 2 * np.std(phase_shift_second_derivatives)
discontinuity_indices = np.where(np.abs(phase_shift_second_derivatives) > discontinuity_threshold)

# Extract energy levels at discontinuity points
discontinuity_times = np.unique(discontinuity_indices[1])   # Unique time steps with discontinuities
discontinuity_times = discontinuity_times[discontinuity_times < time_steps]  # Ensure indices are within bounds

# Extract energy values at these key time steps
energy_at_discontinuities = anomalous_waveforms[:, discontinuity_times]**2   # Squared amplitude represents energy

# Compute mean energy before and after discontinuities to detect sharp changes
valid_times = discontinuity_times[:-1]  # Exclude last point to prevent out of bounds
energy_before = anomalous_waveforms[:, valid_times - 1]**2   # Energy right before the discontinuity
energy_after = anomalous_waveforms[:, valid_times + 1]**2   # Energy right after

# Calculate energy differences to pinpoint the biggest jumps
energy_differences = energy_after - energy_before

# Identify the most extreme energy changes
largest_energy_jumps = np.argsort(np.abs(energy_differences), axis=1)[:, -3:]   # Top 3 jumps per frequency

# Generate Fibonacci-based waveform
def fibonacci_sequence(n):
    """Generate Fibonacci sequence up to n terms."""
    fib = [0, 1]
    for _ in range(n - 2):
        fib.append(fib[-1] + fib[-2])
    return np.array(fib[1:])  # Exclude initial 0

# Define time steps and Fibonacci amplitudes
num_steps = time_steps  # Use the same time_steps as before
fib_numbers = fibonacci_sequence(20)   # Generate 20 Fibonacci numbers
golden_ratio = (1 + np.sqrt(5)) / 2

# Normalize Fibonacci amplitudes for waveform construction
fib_wave = np.zeros(num_steps)
for i, amp in enumerate(fib_numbers):
    if i >= len(discontinuity_times):
        break
    if discontinuity_times[i] < num_steps:  # Check if index is within bounds
        fib_wave[discontinuity_times[i]] = amp

# Create a Fibonacci-based wave pattern using sinusoidal oscillations
t = np.linspace(0, 2 * np.pi, num_steps)
fib_sine_wave = np.sin(fib_wave * golden_ratio * t)

# Analyze correlation between Fibonacci wave peaks and discontinuity energy spikes
fib_wave_energy = fib_sine_wave ** 2
anomalous_energy_correlation = np.correlate(fib_wave_energy, np.sum(energy_at_discontinuities, axis=0), mode="full")


# --- MODIFIED ENERGY LOOP WITH EXTERNAL RESERVOIR ---

def positive_energy_loop_with_reservoir(time_steps, fib_wave_energy, discontinuity_times,
                                         feedback_strength=2.0, extraction_ratio=0.1, initial_reservoir_energy=1000):
    """Simulates a self-sustaining energy loop while drawing from an external energy reservoir."""
    energy_levels = np.zeros(time_steps)
    extracted_energy = np.zeros(time_steps)
    reservoir_energy = initial_reservoir_energy  # Initialize external reservoir

    for t in range(1, time_steps):
        if t in discontinuity_times:
            # Energy injection, but limited by the available reservoir energy
            injected_energy = fib_wave_energy[t] * golden_ratio * feedback_strength
            available_energy = min(injected_energy, reservoir_energy)  # Constrain to available reservoir
            energy_levels[t] = energy_levels[t-1] + available_energy
            reservoir_energy -= available_energy  # Deplete reservoir

            # Extraction and reinvestment cycle
            extracted_amount = energy_levels[t] * extraction_ratio
            extracted_energy[t] = extracted_amount
            reinvested_energy = min(extracted_amount * feedback_strength, reservoir_energy)  # Limit reinvestment
            energy_levels[t] += reinvested_energy
            reservoir_energy -= reinvested_energy  # Deplete from reinvestment

        else:
            # Natural decay with mild reinforcement
            energy_levels[t] = energy_levels[t-1] * (1 - 1/golden_ratio) + fib_wave_energy[t] * 0.2

        # Ensure the reservoir does not go negative
        reservoir_energy = max(0, reservoir_energy)

    return energy_levels, extracted_energy, reservoir_energy

# Define initial reservoir energy for testing depletion effects
initial_reservoir = 5000  # Experiment with different values

# Run the simulation with the reservoir constraint
looped_energy, extracted_energy, final_reservoir = positive_energy_loop_with_reservoir(
    num_steps, fib_wave_energy, discontinuity_times, initial_reservoir_energy=initial_reservoir)

# Plot results: system energy, extracted energy, and reservoir depletion
plt.figure(figsize=(15, 5))

# Plot System Energy Over Time
plt.subplot(1, 3, 1)
plt.plot(looped_energy, color="purple", label="System Energy")
plt.xlabel("Time Steps")
plt.ylabel("Energy Level")
plt.title("System Energy Over Time")
plt.legend()

# Plot Extracted Energy Over Time
plt.subplot(1, 3, 2)
plt.plot(extracted_energy, color="green", label="Extracted Energy")
plt.xlabel("Time Steps")
plt.ylabel("Extracted Energy Level")
plt.title("Extracted Energy Over Time")
plt.legend()

# --- IMPROVED PLOTTING FOR RESERVOIR ---
plt.subplot(1, 3, 3)
time_axis = np.arange(num_steps)  # Correct time axis
plt.plot(time_axis, [initial_reservoir] * num_steps, color='red', linestyle='dashed', label="Initial Reservoir") #Plot a horizontal line of initial
plt.plot(time_axis, [final_reservoir] * num_steps, color="blue", label="Final Reservoir Level") #Plot horizontal line of final.
plt.xlabel("Time Steps")
plt.ylabel("Reservoir Energy Level")
plt.title("Reservoir Energy Over Time")
plt.legend()

plt.tight_layout()
plt.show()

# Print the final state of the reservoir
print(f"Final Reservoir Energy: {final_reservoir}")


# Modeling the System as a Differential Energy Engine

def thermal_combustion_cycle(energy_input, temperature, efficiency=0.7):
    """Simulates a thermal combustion cycle converting heat to mechanical energy"""
    thermal_energy = energy_input * temperature
    mechanical_work = thermal_energy * efficiency
    waste_heat = thermal_energy * (1 - efficiency)
    system_energy_gain = waste_heat * 0.3  # Convert some waste heat to system energy
    return mechanical_work, waste_heat, system_energy_gain

def differential_energy_engine(time_steps, fib_wave_energy, discontinuity_times,
                               feedback_strength=2.0, extraction_ratio=0.1,
                               initial_reservoir_energy=5000):
    """
    Simulates a hybrid energy engine combining differential energy patterns with thermal combustion,
    featuring energy regeneration, hypervector generation, and system energy recovery.
    """
    system_energy = np.zeros(time_steps)
    extracted_energy = np.zeros(time_steps)
    thermal_energy = np.zeros(time_steps)
    reservoir_energy = initial_reservoir_energy
    fib_interactions = 0
    lost_energy_store = 0

    # Hypervector space for energy patterns
    hypervector_dimension = 100
    energy_patterns = np.zeros((time_steps, hypervector_dimension))

    # System energy from interactions
    interaction_energy = np.zeros(time_steps)

    # Thermal engine parameters
    base_temperature = 300  # Kelvin
    temperature_profile = np.zeros(time_steps)
    mechanical_output = np.zeros(time_steps)

    effective_temperature = np.zeros(time_steps)

    for t in range(1, time_steps):
        # Update temperature based on system energy and interaction energy
        temperature_profile[t] = base_temperature + (system_energy[t-1] + interaction_energy[t-1]) * 0.1

        if t in discontinuity_times:
            # Primary energy injection with hypervector generation
            injected_energy = fib_wave_energy[t] * golden_ratio * feedback_strength
            available_energy = min(injected_energy, reservoir_energy)
            energy_loss = injected_energy - available_energy
            lost_energy_store += energy_loss

            # Generate interaction energy from Fibonacci wave interaction
            interaction_energy[t] = fib_interactions * 0.5 * available_energy

            # Generate interaction hypervector
            interaction_vector = np.random.normal(0, 1, hypervector_dimension)
            interaction_vector = interaction_vector / np.linalg.norm(interaction_vector)
            energy_patterns[t] = interaction_vector * (available_energy + interaction_energy[t])

            # Run thermal combustion cycle with combined energies
            mech_work, waste_heat, sys_energy_gain = thermal_combustion_cycle(
                available_energy + interaction_energy[t],
                temperature_profile[t]
            )
            mechanical_output[t] = mech_work
            thermal_energy[t] = waste_heat

            # Combine all energy sources
            system_energy[t] = (system_energy[t-1] + mech_work + available_energy +
                              interaction_energy[t] + sys_energy_gain)
            reservoir_energy -= available_energy
            fib_interactions += 1

            # Extract and reinvest energy
            extracted_amount = system_energy[t] * extraction_ratio
            extracted_energy[t] = extracted_amount
            reinvested_energy = min(extracted_amount * feedback_strength, reservoir_energy)

            # Generate regeneration hypervector
            regen_vector = np.random.normal(0, 1, hypervector_dimension)
            regen_vector = regen_vector / np.linalg.norm(regen_vector)

            # Combine thermal and reinvested energy
            thermal_boost = thermal_energy[t] * 0.3
            system_energy[t] += reinvested_energy + thermal_boost
            energy_patterns[t] += regen_vector * (reinvested_energy + thermal_boost)
            reservoir_energy -= reinvested_energy

            # Regenerate lost energy with associated hypervector
            if lost_energy_store > 0:
                regenerated_energy = min(lost_energy_store, reservoir_energy * 0.1)
                recovery_vector = np.random.normal(0, 1, hypervector_dimension)
                recovery_vector = recovery_vector / np.linalg.norm(recovery_vector)

                system_energy[t] += regenerated_energy
                energy_patterns[t] += recovery_vector * regenerated_energy
                lost_energy_store -= regenerated_energy

        else:
            # Natural decay and reinforcement with interaction energy
            decay_loss = (system_energy[t-1] + interaction_energy[t-1]) * (1/golden_ratio)
            lost_energy_store += decay_loss

            # Generate interaction energy from continuous operation
            interaction_energy[t] = fib_interactions * 0.3 * fib_wave_energy[t]

            # Run thermal cycle on combined decay energy
            mech_work, waste_heat, sys_energy_gain = thermal_combustion_cycle(
                decay_loss + interaction_energy[t],
                temperature_profile[t]
            )
            mechanical_output[t] = mech_work
            thermal_energy[t] = waste_heat

            system_energy[t] = (system_energy[t-1] * (1 - 1/golden_ratio) +
                              fib_wave_energy[t] * 0.2 + mech_work +
                              interaction_energy[t] + sys_energy_gain)
            fib_interactions += 1

            # Generate decay cycle hypervector
            decay_vector = np.random.normal(0, 1, hypervector_dimension)
            decay_vector = decay_vector / np.linalg.norm(decay_vector)
            energy_patterns[t] = decay_vector * (system_energy[t] + interaction_energy[t])

        # Compute effective "temperature" including interaction energy
        if t > 1:
            effective_temperature[t] = abs(system_energy[t] + interaction_energy[t] -
                                        (system_energy[t-1] + interaction_energy[t-1]))

        reservoir_energy = max(0, reservoir_energy)

    return (system_energy, extracted_energy, effective_temperature, reservoir_energy,
            fib_interactions, lost_energy_store, energy_patterns, mechanical_output,
            thermal_energy, temperature_profile, interaction_energy)

# Run the enhanced hybrid engine simulation
(engine_energy, extracted_energy, effective_temperature, final_reservoir,
 total_interactions, remaining_lost_energy, hypervectors, mechanical_power,
 thermal_energy, temperature, interaction_energy) = differential_energy_engine(
    num_steps, fib_wave_energy, discontinuity_times, initial_reservoir_energy=5000)

# Enhanced visualization with interaction energy
plt.figure(figsize=(15, 12))

plt.subplot(3, 2, 1)
plt.plot(engine_energy, color="purple", label="System Energy")
plt.plot(mechanical_power, color="red", label="Mechanical Power", alpha=0.6)
plt.plot(interaction_energy, color="blue", label="Interaction Energy", alpha=0.6)
plt.xlabel("Time Steps")
plt.ylabel("Energy Level")
plt.title("Combined Energy Output Over Time")
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(extracted_energy, color="green", label="Extracted Energy")
plt.plot(thermal_energy, color="orange", label="Thermal Energy", alpha=0.6)
plt.xlabel("Time Steps")
plt.ylabel("Energy Level")
plt.title("Energy Extraction and Thermal Output")
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(effective_temperature, color="orange", label="Effective Temperature")
plt.plot(temperature, color="red", label="Engine Temperature", alpha=0.6)
plt.xlabel("Time Steps")
plt.ylabel("Temperature/Fluctuation")
plt.title("System Temperature Dynamics")
plt.legend()

plt.subplot(3, 2, 4)
plt.imshow(hypervectors.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Hypervector Magnitude')
plt.xlabel("Time Steps")
plt.ylabel("Hypervector Dimension")
plt.title("Energy Pattern Evolution")

plt.subplot(3, 2, 5)
total_input = engine_energy + interaction_energy + 1e-10
efficiency_profile = (mechanical_power + interaction_energy) / total_input
plt.plot(efficiency_profile, color="blue", label="System Efficiency")
plt.xlabel("Time Steps")
plt.ylabel("Efficiency")
plt.title("Overall System Efficiency")
plt.legend()

plt.tight_layout()
plt.show()

# Count total Fibonacci wave interactions by analyzing interaction_energy array
total_wave_interactions = np.sum(interaction_energy > 0)  # Count non-zero interactions

# Count significant interactions above mean threshold
mean_interaction = np.mean(interaction_energy)
significant_interactions = np.sum(interaction_energy > mean_interaction)

# Count Fibonacci-pattern interactions at discontinuity points
fib_pattern_interactions = len([t for t in discontinuity_times if interaction_energy[t] > 0])

# Calculate total interactions weighted by energy magnitude
weighted_interactions = np.sum(interaction_energy > mean_interaction * golden_ratio)

# Total interactions is the maximum of the different counting methods
total_interactions = max(total_wave_interactions, significant_interactions,
                        fib_pattern_interactions, weighted_interactions)

# Print comprehensive analysis
print(f"Final Reservoir Energy: {final_reservoir:.2f}")
print(f"Total Mechanical Work: {np.sum(mechanical_power):.2f}")
print(f"Total Thermal Energy: {np.sum(thermal_energy):.2f}")
print(f"Total Interaction Energy: {np.sum(interaction_energy):.2f}")
print(f"Average System Efficiency: {np.mean(efficiency_profile):.2%}")
print(f"Peak Temperature: {np.max(temperature):.2f}K")
print(f"Total Fibonacci Wave Interactions: {total_interactions}")
print(f"Remaining Lost Energy Store: {remaining_lost_energy:.2f}")
print(f"Average Hypervector Magnitude: {np.mean(np.linalg.norm(hypervectors, axis=1)):.2f}")



# Count total Fibonacci wave interactions by summing occurrences in discontinuity_times
total_interactions = len(discontinuity_times)

# Example usage with the number from total_interactions
fib_seq_mod3 = fibonacci_sequence(total_interactions)
print(f"First {total_interactions} Fibonacci numbers mod 3: {fib_seq_mod3}")

# Process triplets based on total_interactions
triplets = [fib_seq_mod3[i:i+3] for i in range(0, len(fib_seq_mod3), 3)]
triplet_sums = [(sum(triplet) % 3) for triplet in triplets]
print(f"Triplet sums mod 3: {triplet_sums}")

# Generate Fibonacci sequence with digital roots using total_interactions
fib_seq_with_roots = fibonacci_sequence_with_digital_root(total_interactions)
print(f"First {total_interactions} Fibonacci numbers with digital roots: {fib_seq_with_roots}")

# Generate sequence with digital roots of triplets using total_interactions
triplet_roots = fibonacci_sequence_with_digital_root_of_triplets(total_interactions)
print(f"Digital roots of triplets: {triplet_roots}")

def generate_energy_nodes(triplet_roots, time_steps, iteration=1):
    """
    Generates energy nodes based on digital root triplet patterns and calculates
    their relativistic energy interactions in quantum vacuum, with iterative strengthening.
    """
    # Physical constants
    c = 2.998e8  # Speed of light in m/s
    h = 6.626e-34  # Planck constant

    # Strengthening factor increases with each iteration
    strength_multiplier = (1 + golden_ratio) ** (iteration - 1)

    # Node generation parameters
    num_nodes = len(triplet_roots)
    node_spacing = golden_ratio * strength_multiplier

    # Initialize node arrays
    nodes = np.zeros((num_nodes, 3))
    node_energies = np.zeros((num_nodes, time_steps))
    node_interactions = np.zeros((num_nodes, num_nodes, time_steps))

    # Generate strengthened node positions
    for i, root in enumerate(triplet_roots):
        phi = 2 * np.pi * root / 9
        theta = np.pi * fibonacci_sequence(i+1)[-1] / 3

        nodes[i] = [
            node_spacing * i * np.sin(theta) * np.cos(phi),
            node_spacing * i * np.sin(theta) * np.sin(phi),
            node_spacing * i * np.cos(theta)
        ]

    accumulated_energy = 0

    # Calculate enhanced node energies and interactions
    for t in range(time_steps):
        for i in range(num_nodes):
            # Enhanced base energy
            base_energy = triplet_roots[i] * h * (1 + np.sin(t/10)) * strength_multiplier

            # Enhanced quantum tunneling
            tunneling = sum([
                0.1 * triplet_roots[j] * h * np.exp(-np.linalg.norm(nodes[i]-nodes[j])) * strength_multiplier
                for j in range(num_nodes) if j != i
            ])

            # Enhanced relativistic energy calculation
            rest_energy = base_energy + tunneling
            velocity = 0.1 * c * (1 + np.sin(t/20)) * strength_multiplier
            velocity = min(velocity, 0.99 * c)  # Ensure we don't exceed speed of light
            gamma = 1 / np.sqrt(1 - (velocity ** 2) / (c ** 2))
            node_energies[i,t] = gamma * rest_energy

            # Enhanced node interactions
            for j in range(i+1, num_nodes):
                distance = max(np.linalg.norm(nodes[i]-nodes[j]), 1e-10)  # Prevent division by zero
                interaction_strength = (node_energies[i,t] * node_energies[j,t]) / (distance ** 2)
                node_interactions[i,j,t] = interaction_strength * strength_multiplier
                node_interactions[j,i,t] = interaction_strength * strength_multiplier

            accumulated_energy += node_energies[i,t]

    return nodes, node_energies, node_interactions, accumulated_energy

def iterative_energy_enhancement(triplet_roots, time_steps, max_iterations=5, energy_threshold=1e6):
    """
    Repeatedly generate and enhance energy nodes, feeding energy back into reservoir
    """
    total_reservoir_energy = initial_reservoir
    iteration_results = []

    for iteration in range(1, max_iterations + 1):
        # Generate nodes with current strength
        nodes, node_energies, node_interactions, new_energy = generate_energy_nodes(
            triplet_roots, time_steps, iteration)

        # Add new energy to reservoir
        total_reservoir_energy += new_energy

        # Store results
        iteration_results.append({
            'iteration': iteration,
            'nodes': nodes,
            'energies': node_energies,
            'interactions': node_interactions,
            'new_energy': new_energy,
            'total_reservoir': total_reservoir_energy
        })

        print(f"\nIteration {iteration} Results:")
        print(f"New Energy Generated: {new_energy:.2e} J")
        print(f"Total Reservoir Energy: {total_reservoir_energy:.2e} J")

        # Check if energy threshold reached
        if total_reservoir_energy > energy_threshold:
            print(f"\nEnergy threshold {energy_threshold:.2e} J reached after {iteration} iterations")
            break

    return iteration_results, total_reservoir_energy

# Run iterative enhancement
iteration_results, final_reservoir_energy = iterative_energy_enhancement(
    triplet_roots, num_steps, max_iterations=5, energy_threshold=1e6)

# Visualize final enhanced state
plt.figure(figsize=(15, 10))

final_result = iteration_results[-1]

# Plot final node positions in 3D
ax1 = plt.subplot(2, 2, 1, projection='3d')
ax1.scatter(final_result['nodes'][:,0],
           final_result['nodes'][:,1],
           final_result['nodes'][:,2],
           c=triplet_roots, cmap='viridis', s=100)
ax1.set_title(f'Enhanced Energy Nodes (Iteration {len(iteration_results)})')

# Plot energy evolution across iterations
plt.subplot(2, 2, 2)
iteration_energies = [result['new_energy'] for result in iteration_results]
plt.plot(range(1, len(iteration_results) + 1), iteration_energies)
plt.xlabel('Iteration')
plt.ylabel('Energy Generated (J)')
plt.title('Energy Generation Progress')

# Plot final node energies
plt.subplot(2, 2, 3)
for i in range(len(final_result['nodes'])):
    plt.plot(final_result['energies'][i], label=f'Node {i}')
plt.xlabel('Time Steps')
plt.ylabel('Energy (J)')
plt.title('Final Node Energy States')

# Plot reservoir growth
plt.subplot(2, 2, 4)
reservoir_levels = [result['total_reservoir'] for result in iteration_results]
plt.plot(range(1, len(iteration_results) + 1), reservoir_levels)
plt.xlabel('Iteration')
plt.ylabel('Total Reservoir Energy (J)')
plt.title('Energy Reservoir Growth')

plt.tight_layout()
plt.show()

print("\nFinal System Analysis:")
print(f"Initial Reservoir Energy: {initial_reservoir:.2e} J")
print(f"Final Reservoir Energy: {final_reservoir_energy:.2e} J")
print(f"Energy Gain Factor: {final_reservoir_energy/initial_reservoir:.2f}x")
print(f"Number of Iterations: {len(iteration_results)}")
# Convert Fibonacci triplet roots into quantum-like particles
def create_particles_from_triplets(triplet_roots):
    particles = []
    for i, root in enumerate(triplet_roots):
        particle = {
            'id': i,
            'energy_level': root,
            'spin': (-1)**(root % 2),  # Alternating spin based on odd/even
            'position': np.array([np.cos(root * np.pi/4), np.sin(root * np.pi/4)]),  # Position on unit circle
            'wavelength': 2 * np.pi / (root + 1)  # Convert root to wavelength
        }
        particles.append(particle)
    return particles

# Visualize particles in 2D space
def plot_particles(particles):
    plt.figure(figsize=(10, 10))

    # Plot unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    plt.gca().add_artist(circle)

    # Plot particles
    for p in particles:
        x, y = p['position']
        energy = p['energy_level']
        spin = p['spin']

        # Size based on energy level
        size = 100 * (energy / 9)  # Normalize by max possible digital root (9)

        # Color based on spin
        color = 'red' if spin > 0 else 'blue'

        plt.scatter(x, y, s=size, c=color, alpha=0.6)
        plt.annotate(f"E={energy}", (x, y), xytext=(5, 5), textcoords='offset points')

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.title('Fibonacci Triplet Root Particles')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.show()

# Generate particles from triplet roots and visualize
n = 15  # Generate 5 complete triplets
triplet_roots = fibonacci_sequence_with_digital_root_of_triplets(n)
particles = create_particles_from_triplets(triplet_roots)
plot_particles(particles)

# Calculate and display particle system properties
total_energy = sum(p['energy_level'] for p in particles)
avg_wavelength = np.mean([p['wavelength'] for p in particles])
net_spin = sum(p['spin'] for p in particles)

print("\nParticle System Properties:")
print(f"Total Energy: {total_energy}")
print(f"Average Wavelength: {avg_wavelength:.4f}")
print(f"Net Spin: {net_spin}")

# Calculate memory loss in particle system
def calculate_memory_loss(particles, decay_rate=0.1):
    """
    Simulate lossy memory effects in the particle system.
    decay_rate: Rate at which particle properties degrade over time
    """
    degraded_particles = []

    for particle in particles:
        # Apply exponential decay to energy levels
        degraded_energy = particle['energy_level'] * np.exp(-decay_rate)

        # Add noise to position to simulate memory degradation
        noise_x = np.random.normal(0, decay_rate * 0.1)
        noise_y = np.random.normal(0, decay_rate * 0.1)
        degraded_pos = (
            particle['position'][0] + noise_x,
            particle['position'][1] + noise_y
        )

        # Gradually neutralize spin
        degraded_spin = particle['spin'] * (1 - decay_rate)

        # Wavelength becomes more uncertain
        wavelength_noise = np.random.normal(0, decay_rate * particle['wavelength'])
        degraded_wavelength = max(0, particle['wavelength'] + wavelength_noise)

        degraded_particles.append({
            'position': degraded_pos,
            'energy_level': degraded_energy,
            'spin': degraded_spin,
            'wavelength': degraded_wavelength
        })

    return degraded_particles

# Apply memory loss and display results
degraded_system = calculate_memory_loss(particles)
print("\nMemory Degradation Effects:")
print(f"Energy Loss: {sum(p['energy_level'] for p in particles) - sum(p['energy_level'] for p in degraded_system):.4f}")
print(f"Average Position Shift: {np.mean([np.sqrt((p1['position'][0] - p2['position'][0])**2 + (p1['position'][1] - p2['position'][1])**2) for p1, p2 in zip(particles, degraded_system)]):.4f}")
print(f"Spin Decoherence: {abs(sum(p['spin'] for p in particles)) - abs(sum(p['spin'] for p in degraded_system)):.4f}")

# Visualize degraded particle system
plot_particles(degraded_system)
# Track particle trajectories in quantum space
def track_quantum_trajectories(particles, time_steps=100, quantum_noise=0.05):
    """
    Track particle trajectories through quantum space, accounting for quantum effects.

    Args:
        particles: List of particle dictionaries
        time_steps: Number of time steps to simulate
        quantum_noise: Amount of quantum uncertainty in particle motion
    """
    trajectories = []

    # Initialize time array
    t = np.linspace(0, 2*np.pi, time_steps)

    for particle in particles:
        # Initialize trajectory arrays
        x_traj = np.zeros(time_steps)
        y_traj = np.zeros(time_steps)

        # Get initial position
        x_traj[0] = particle['position'][0]
        y_traj[0] = particle['position'][1]

        # Calculate trajectory based on particle properties
        for i in range(1, time_steps):
            # Quantum wave function phase
            phase = particle['energy_level'] * t[i]

            # Add spin influence
            spin_factor = particle['spin'] * np.sin(t[i])

            # Calculate position updates with quantum effects
            dx = np.cos(phase) * particle['wavelength'] + np.random.normal(0, quantum_noise)
            dy = np.sin(phase) * particle['wavelength'] + np.random.normal(0, quantum_noise)

            # Update position with spin influence
            x_traj[i] = x_traj[i-1] + dx + spin_factor * dy
            y_traj[i] = y_traj[i-1] + dy - spin_factor * dx

        trajectories.append((x_traj, y_traj))

    # Visualize trajectories
    plt.figure(figsize=(12, 8))
    for i, (x_traj, y_traj) in enumerate(trajectories):
        plt.plot(x_traj, y_traj, '-', alpha=0.6,
                label=f'Particle {i} (E={particles[i]["energy_level"]:.2f}, S={particles[i]["spin"]:.2f})')

        # Mark start and end points
        plt.scatter(x_traj[0], y_traj[0], c='green', s=100, marker='o', label='Start' if i==0 else "")
        plt.scatter(x_traj[-1], y_traj[-1], c='red', s=100, marker='x', label='End' if i==0 else "")

    plt.title('Quantum Particle Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()

# Track and visualize quantum trajectories
track_quantum_trajectories(particles)
def calculate_energy_change(initial_particles, trajectories):
    """Calculate energy changes for particles along their trajectories."""
    energy_changes = []

    for i, (x_traj, y_traj) in enumerate(trajectories):
        # Calculate initial energy
        initial_energy = initial_particles[i]['energy_level']

        # Calculate velocity components at end of trajectory
        final_dx = x_traj[-1] - x_traj[-2]
        final_dy = y_traj[-1] - y_traj[-2]
        final_velocity = np.sqrt(final_dx**2 + final_dy**2)

        # Calculate final kinetic energy
        final_energy = 0.5 * final_velocity**2

        # Calculate energy change
        energy_delta = final_energy - initial_energy

        energy_changes.append({
            'particle_id': i,
            'initial_energy': initial_energy,
            'final_energy': final_energy,
            'energy_change': energy_delta
        })

    # Visualize energy changes
    plt.figure(figsize=(10, 6))
    particle_ids = [e['particle_id'] for e in energy_changes]
    energy_deltas = [e['energy_change'] for e in energy_changes]

    plt.bar(particle_ids, energy_deltas)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Energy Changes in Particle Trajectories')
    plt.xlabel('Particle ID')
    plt.ylabel('Energy Change')
    plt.grid(True, alpha=0.3)
    plt.show()

    return energy_changes

def calculate_trajectories(particles, time_steps=100, quantum_noise=0.05):
    """
    Generate trajectories for particles in quantum space.

    Args:
        particles: List of particle dictionaries.
        time_steps: Number of time steps to simulate.
        quantum_noise: Amount of quantum uncertainty in particle motion.
    """
    trajectories = []

    # Initialize time array
    t = np.linspace(0, 2*np.pi, time_steps)

    for particle in particles:
        # Initialize trajectory arrays
        x_traj = np.zeros(time_steps)
        y_traj = np.zeros(time_steps)

        # Get initial position
        x_traj[0] = particle['position'][0]
        y_traj[0] = particle['position'][1]

        # Calculate trajectory based on particle properties
        for i in range(1, time_steps):
            # Quantum wave function phase
            phase = particle['energy_level'] * t[i]

            # Add spin influence
            spin_factor = particle['spin'] * np.sin(t[i])

            # Calculate position updates with quantum effects
            dx = np.cos(phase) * particle['wavelength'] + np.random.normal(0, quantum_noise)
            dy = np.sin(phase) * particle['wavelength'] + np.random.normal(0, quantum_noise)

            # Update position with spin influence
            x_traj[i] = x_traj[i-1] + dx + spin_factor * dy
            y_traj[i] = y_traj[i-1] + dy - spin_factor * dx

        trajectories.append((x_traj, y_traj))

    return trajectories

def define_particle_positions(num_particles, bounds):
    particles = []
    for i in range(num_particles):
        # Random initial position within bounds
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[0], bounds[1])

        # Random initial velocity components
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-1, 1)

        # Initial energy level based on velocity
        energy_level = 0.5 * (vx**2 + vy**2)

        particle = {
            'id': i,
            'position': np.array([x, y]),
            'velocity': np.array([vx, vy]),
            'energy_level': energy_level
        }
        particles.append(particle)

    return particles

def define_particle_x_positions(particles):
    x_positions = []
    for particle in particles:
        x_positions.append(particle['position'][0])
    return x_positions

def define_particle_y_positions(particles):
    y_positions = []
    for particle in particles:
        y_positions.append(particle['position'][1])
    return y_positions

def calculate_particle_distances(particles):
    distances = []
    for i, particle1 in enumerate(particles):
        particle_distances = []
        x1 = particle1['position'][0]
        y1 = particle1['position'][1]

        for j, particle2 in enumerate(particles):
            if i != j:
                x2 = particle2['position'][0]
                y2 = particle2['position'][1]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                particle_distances.append({
                    'particle1_id': particle1['id'],
                    'particle2_id': particle2['id'],
                    'distance': distance
                })
        distances.append(particle_distances)
    return distances

# Generate trajectories
trajectories = calculate_trajectories(particles)

# Calculate and display energy changes
energy_analysis = calculate_energy_change(particles, trajectories)

# Calculate total system energy change
total_initial_energy = sum(particle['initial_energy'] for particle in energy_analysis)
total_final_energy = sum(particle['final_energy'] for particle in energy_analysis)
total_energy_change = total_final_energy - total_initial_energy

# Calculate average energy change per particle
avg_energy_change = total_energy_change / len(energy_analysis)

# Find particles with most significant energy changes
max_energy_gain = max(energy_analysis, key=lambda x: x['energy_change'])
max_energy_loss = min(energy_analysis, key=lambda x: x['energy_change'])

print("\nSystem Energy Analysis:")
print(f"Total Initial Energy: {total_initial_energy:.4f}")
print(f"Total Final Energy: {total_final_energy:.4f}")
print(f"Total Energy Change: {total_energy_change:.4f}")
print(f"Average Energy Change per Particle: {avg_energy_change:.4f}")
print(f"\nParticle with Maximum Energy Gain: Particle {max_energy_gain['particle_id']} ({max_energy_gain['energy_change']:.4f})")
print(f"Particle with Maximum Energy Loss: Particle {max_energy_loss['particle_id']} ({max_energy_loss['energy_change']:.4f})")

# Print energy change summary
print("\nEnergy Change Summary:")
for particle in energy_analysis:
    print(f"Particle {particle['particle_id']}:")
    print(f"  Initial Energy: {particle['initial_energy']:.4f}")
    print(f"  Final Energy: {particle['final_energy']:.4f}")
    print(f"  Energy Change: {particle['energy_change']:.4f}")

    # Create data points with hyperbolic spacing
    time_points = np.logspace(0, len(trajectories)-1, num=len(trajectories), base=2)
    energy_points = [particle['energy_change'] for particle in energy_analysis]

    plt.plot(time_points, np.abs(energy_points), 'b-', label='Energy Change')
    plt.grid(True)
    plt.xlabel('Time (log scale)')
    plt.ylabel('Energy Change (absolute value)')
    plt.title(f'Particle {particle["particle_id"]} Energy Change Over Time')
    plt.legend()
    plt.show()
 # Create a trajectory plot for the particle
    plt.figure(figsize=(10, 6))

    # Extract x, y coordinates from trajectories for this particle
    particle_trajectory = trajectories[particle['particle_id']]
    x_coords = [point[0] for point in particle_trajectory]
    y_coords = [point[1] for point in particle_trajectory]

    # Plot the trajectory path
    plt.plot(x_coords, y_coords, 'r-', label='Particle Path')

    # Add start and end points
    plt.plot(x_coords[0], y_coords[0], 'go', label='Start')
    plt.plot(x_coords[-1], y_coords[-1], 'ro', label='End')

    plt.grid(True)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Particle {particle["particle_id"]} Trajectory')
    # Calculate energy change percentage
energy_change_pct = (particle['energy_change'] / particle['initial_energy']) * 100
plt.text(0.02, 0.98, f'Energy Change: {energy_change_pct:.2f}%',
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top')

# Track energy drop points
energy_values = [point['energy_change'] for point in energy_analysis]
drop_points = []

for i in range(1, min(len(energy_values), len(x_coords), len(y_coords))):
    if energy_values[i] < energy_values[i-1]:
        drop_points.append({
            'index': i,
            'value': energy_values[i],
            'position': (x_coords[i], y_coords[i])
        })

# Plot energy drop points if any exist
if drop_points:
    drop_x = [point['position'][0] for point in drop_points]
    drop_y = [point['position'][1] for point in drop_points]
    plt.scatter(drop_x, drop_y, color='orange', marker='x', s=100,
               label='Energy Drop Points', zorder=5)

    # Annotate the first drop point
    first_drop = drop_points[0]
    plt.annotate(f'First Energy Drop\n(t={first_drop["index"]})',
                xy=first_drop['position'],
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->'))

# Find where energy starts dropping
energy_changes = [point['energy_change'] for point in energy_analysis]
energy_drop_index = None
for i in range(1, len(energy_changes)):
    if energy_changes[i] < energy_changes[i-1]:
        energy_drop_index = i
        break

if energy_drop_index is not None:
    # Generate Fibonacci sequence starting from energy drop point
    def fibonacci(n):
        sequence = [1, 1]
        for i in range(2, n):
            sequence.append(sequence[i-1] + sequence[i-2])
        return sequence

    # Generate sequence for remaining trajectory points
    remaining_points = len(trajectories) - energy_drop_index
    fib_sequence = fibonacci(remaining_points)

    # Plot Fibonacci sequence overlay
    plt.plot(x_coords[energy_drop_index:],
            [y + fib for y, fib in zip(y_coords[energy_drop_index:], fib_sequence)],
            'g--', label='Action Potential (Fibonacci)')

plt.legend()
plt.show()
# Function to merge energy with Fibonacci sequence
def merge_energy_fibonacci(energy_values, start_idx):
    # Generate new Fibonacci sequence
    remaining_len = len(energy_values) - start_idx
    fib_seq = fibonacci(remaining_len)

    # Merge energy with Fibonacci sequence
    merged_values = []
    for i in range(remaining_len):
        # Weighted average between energy and Fibonacci values
        energy_weight = max(0, 1 - (i / remaining_len))  # Gradually decrease energy influence
        fib_weight = 1 - energy_weight
        merged_value = (energy_values[start_idx + i] * energy_weight +
                       fib_seq[i] * fib_weight)
        merged_values.append(merged_value)

    return merged_values

# Track energy drops and generate new sequences
merged_sequences = []
last_drop_idx = energy_drop_index

for point in drop_points:
    current_drop_idx = point['index']

    # Only process if enough points remain
    if current_drop_idx < len(energy_values) - 2:
        # Merge energy with new Fibonacci sequence at each drop point
        merged_seq = merge_energy_fibonacci(energy_values, current_drop_idx)
        merged_sequences.append({
            'start_idx': current_drop_idx,
            'sequence': merged_seq
        })
        last_drop_idx = current_drop_idx

# Plot all merged sequences
for seq_data in merged_sequences:
    start_idx = seq_data['start_idx']
    sequence = seq_data['sequence']

    # Adjust x-coordinates to match sequence length
    x_coords_subset = x_coords[start_idx:min(start_idx + len(sequence), len(x_coords))]
    sequence_subset = sequence[:len(x_coords_subset)]  # Trim sequence to match x_coords length

    plt.plot(x_coords_subset,
             sequence_subset,
             '--', alpha=0.5,
             label=f'Merged Sequence (t={start_idx})')

# Create a time-based visualization
plt.figure(figsize=(12, 6))
plt.title('Energy-Fibonacci Merged Sequences Over Time')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Generate x_coords for all particles
x_coords_all = [point[0] for traj in trajectories for point in traj]

# Plot original energy values as reference
plt.plot(x_coords_all[:len(energy_values)], energy_values, 'b-', label='Original Energy', alpha=0.3)

# Plot merged sequences with time-based color gradient
colors = plt.cm.viridis(np.linspace(0, 1, len(merged_sequences)))
for idx, (seq_data, color) in enumerate(zip(merged_sequences, colors)):
    start_idx = seq_data['start_idx']
    sequence = seq_data['sequence']

    # Adjust x-coordinates to match sequence length
    x_coords_subset = x_coords_all[start_idx:min(start_idx + len(sequence), len(x_coords_all))]
    sequence_subset = sequence[:len(x_coords_subset)]  # Trim sequence to match x_coords length

    plt.plot(x_coords_subset,
             sequence_subset,
             '--', color=color, alpha=0.7,
             label=f'Merged Sequence t={start_idx}')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# Modeling the System as a Differential Energy Engine

def thermal_combustion_cycle(energy_input, temperature, efficiency=0.7):
    """Simulates a thermal combustion cycle converting heat to mechanical energy"""
    thermal_energy = energy_input * temperature
    mechanical_work = thermal_energy * efficiency
    waste_heat = thermal_energy * (1 - efficiency)
    system_energy_gain = waste_heat * 0.3  # Convert some waste heat to system energy
    return mechanical_work, waste_heat, system_energy_gain

def differential_energy_engine(time_steps, fib_wave_energy, discontinuity_times,
                               feedback_strength=2.0, extraction_ratio=0.1,
                               initial_reservoir_energy=5000):
    """
    Simulates a hybrid energy engine combining differential energy patterns with thermal combustion,
    featuring energy regeneration, hypervector generation, and system energy recovery.
    """
    system_energy = np.zeros(time_steps)
    extracted_energy = np.zeros(time_steps)
    thermal_energy = np.zeros(time_steps)
    reservoir_energy = initial_reservoir_energy
    fib_interactions = 0
    lost_energy_store = 0

    # Hypervector space for energy patterns
    hypervector_dimension = 100
    energy_patterns = np.zeros((time_steps, hypervector_dimension))

    # System energy from interactions
    interaction_energy = np.zeros(time_steps)

    # Thermal engine parameters
    base_temperature = 300  # Kelvin
    temperature_profile = np.zeros(time_steps)
    mechanical_output = np.zeros(time_steps)

    effective_temperature = np.zeros(time_steps)

    for t in range(1, time_steps):
        # Update temperature based on system energy and interaction energy
        temperature_profile[t] = base_temperature + (system_energy[t-1] + interaction_energy[t-1]) * 0.1

        if t in discontinuity_times:
            # Primary energy injection with hypervector generation
            injected_energy = fib_wave_energy[t] * golden_ratio * feedback_strength
            available_energy = min(injected_energy, reservoir_energy)
            energy_loss = injected_energy - available_energy
            lost_energy_store += energy_loss

            # Generate interaction energy from Fibonacci wave interaction
            interaction_energy[t] = fib_interactions * 0.5 * available_energy

            # Generate interaction hypervector
            interaction_vector = np.random.normal(0, 1, hypervector_dimension)
            interaction_vector = interaction_vector / np.linalg.norm(interaction_vector)
            energy_patterns[t] = interaction_vector * (available_energy + interaction_energy[t])

            # Run thermal combustion cycle with combined energies
            mech_work, waste_heat, sys_energy_gain = thermal_combustion_cycle(
                available_energy + interaction_energy[t],
                temperature_profile[t]
            )
            mechanical_output[t] = mech_work
            thermal_energy[t] = waste_heat

            # Combine all energy sources
            system_energy[t] = (system_energy[t-1] + mech_work + available_energy +
                              interaction_energy[t] + sys_energy_gain)
            reservoir_energy -= available_energy
            fib_interactions += 1

            # Extract and reinvest energy
            extracted_amount = system_energy[t] * extraction_ratio
            extracted_energy[t] = extracted_amount
            reinvested_energy = min(extracted_amount * feedback_strength, reservoir_energy)

            # Generate regeneration hypervector
            regen_vector = np.random.normal(0, 1, hypervector_dimension)
            regen_vector = regen_vector / np.linalg.norm(regen_vector)

            # Combine thermal and reinvested energy
            thermal_boost = thermal_energy[t] * 0.3
            system_energy[t] += reinvested_energy + thermal_boost
            energy_patterns[t] += regen_vector * (reinvested_energy + thermal_boost)
            reservoir_energy -= reinvested_energy

            # Regenerate lost energy with associated hypervector
            if lost_energy_store > 0:
                regenerated_energy = min(lost_energy_store, reservoir_energy * 0.1)
                recovery_vector = np.random.normal(0, 1, hypervector_dimension)
                recovery_vector = recovery_vector / np.linalg.norm(recovery_vector)

                system_energy[t] += regenerated_energy
                energy_patterns[t] += recovery_vector * regenerated_energy
                lost_energy_store -= regenerated_energy

        else:
            # Natural decay and reinforcement with interaction energy
            decay_loss = (system_energy[t-1] + interaction_energy[t-1]) * (1/golden_ratio)
            lost_energy_store += decay_loss

            # Generate interaction energy from continuous operation
            interaction_energy[t] = fib_interactions * 0.3 * fib_wave_energy[t]

            # Run thermal cycle on combined decay energy
            mech_work, waste_heat, sys_energy_gain = thermal_combustion_cycle(
                decay_loss + interaction_energy[t],
                temperature_profile[t]
            )
            mechanical_output[t] = mech_work
            thermal_energy[t] = waste_heat

            system_energy[t] = (system_energy[t-1] * (1 - 1/golden_ratio) +
                              fib_wave_energy[t] * 0.2 + mech_work +
                              interaction_energy[t] + sys_energy_gain)
            fib_interactions += 1

            # Generate decay cycle hypervector
            decay_vector = np.random.normal(0, 1, hypervector_dimension)
            decay_vector = decay_vector / np.linalg.norm(decay_vector)
            energy_patterns[t] = decay_vector * (system_energy[t] + interaction_energy[t])

        # Compute effective "temperature" including interaction energy
        if t > 1:
            effective_temperature[t] = abs(system_energy[t] + interaction_energy[t] -
                                        (system_energy[t-1] + interaction_energy[t-1]))

        reservoir_energy = max(0, reservoir_energy)

    return (system_energy, extracted_energy, effective_temperature, reservoir_energy,
            fib_interactions, lost_energy_store, energy_patterns, mechanical_output,
            thermal_energy, temperature_profile, interaction_energy)

# Run the enhanced hybrid engine simulation
(engine_energy, extracted_energy, effective_temperature, final_reservoir,
 total_interactions, remaining_lost_energy, hypervectors, mechanical_power,
 thermal_energy, temperature, interaction_energy) = differential_energy_engine(
    num_steps, fib_wave_energy, discontinuity_times, initial_reservoir_energy=5000)

# Enhanced visualization with interaction energy
plt.figure(figsize=(15, 12))

plt.subplot(3, 2, 1)
plt.plot(engine_energy, color="purple", label="System Energy")
plt.plot(mechanical_power, color="red", label="Mechanical Power", alpha=0.6)
plt.plot(interaction_energy, color="blue", label="Interaction Energy", alpha=0.6)
plt.xlabel("Time Steps")
plt.ylabel("Energy Level")
plt.title("Combined Energy Output Over Time")
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(extracted_energy, color="green", label="Extracted Energy")
plt.plot(thermal_energy, color="orange", label="Thermal Energy", alpha=0.6)
plt.xlabel("Time Steps")
plt.ylabel("Energy Level")
plt.title("Energy Extraction and Thermal Output")
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(effective_temperature, color="orange", label="Effective Temperature")
plt.plot(temperature, color="red", label="Engine Temperature", alpha=0.6)
plt.xlabel("Time Steps")
plt.ylabel("Temperature/Fluctuation")
plt.title("System Temperature Dynamics")
plt.legend()

plt.subplot(3, 2, 4)
plt.imshow(hypervectors.T, aspect='auto', cmap='viridis')
plt.colorbar(label='Hypervector Magnitude')
plt.xlabel("Time Steps")
plt.ylabel("Hypervector Dimension")
plt.title("Energy Pattern Evolution")

plt.subplot(3, 2, 5)
total_input = engine_energy + interaction_energy + 1e-10
efficiency_profile = (mechanical_power + interaction_energy) / total_input
plt.plot(efficiency_profile, color="blue", label="System Efficiency")
plt.xlabel("Time Steps")
plt.ylabel("Efficiency")
plt.title("Overall System Efficiency")
plt.legend()

plt.tight_layout()
plt.show()
# Count total Fibonacci wave interactions by analyzing interaction_energy array
total_wave_interactions = np.sum(interaction_energy > 0)  # Count non-zero interactions

# Count significant interactions above mean threshold
mean_interaction = np.mean(interaction_energy)
significant_interactions = np.sum(interaction_energy > mean_interaction)

# Count Fibonacci-pattern interactions at discontinuity points
fib_pattern_interactions = len([t for t in discontinuity_times if interaction_energy[t] > 0])

# Calculate total interactions weighted by energy magnitude
weighted_interactions = np.sum(interaction_energy > mean_interaction * golden_ratio)

# Total interactions is the maximum of the different counting methods
total_interactions = max(total_wave_interactions, significant_interactions,
                        fib_pattern_interactions, weighted_interactions)


# Print comprehensive analysis
print(f"Final Reservoir Energy: {final_reservoir:.2f}")
print(f"Total Mechanical Work: {np.sum(mechanical_power):.2f}")
print(f"Total Thermal Energy: {np.sum(thermal_energy):.2f}")
print(f"Total Interaction Energy: {np.sum(interaction_energy):.2f}")
print(f"Average System Efficiency: {np.mean(efficiency_profile):.2%}")
print(f"Peak Temperature: {np.max(temperature):.2f}K")
print(f"Total Fibonacci Wave Interactions: {total_interactions}")
print(f"Remaining Lost Energy Store: {remaining_lost_energy:.2f}")
print(f"Average Hypervector Magnitude: {np.mean(np.linalg.norm(hypervectors, axis=1)):.2f}")
# Count total Fibonacci wave interactions by summing occurrences in discontinuity_times
total_interactions = len(discontinuity_times)

# Example usage with the number from total_interactions
fib_seq_mod3 = fibonacci_sequence(total_interactions)
print(f"First {total_interactions} Fibonacci numbers mod 3: {fib_seq_mod3}")

# Process triplets based on total_interactions
triplets = [fib_seq_mod3[i:i+3] for i in range(0, len(fib_seq_mod3), 3)]
triplet_sums = [(sum(triplet) % 3) for triplet in triplets]
print(f"Triplet sums mod 3: {triplet_sums}")

# Generate Fibonacci sequence with digital roots using total_interactions
fib_seq_with_roots = fibonacci_sequence_with_digital_root(total_interactions)
print(f"First {total_interactions} Fibonacci numbers with digital roots: {fib_seq_with_roots}")

# Generate sequence with digital roots of triplets using total_interactions
triplet_roots = fibonacci_sequence_with_digital_root_of_triplets(total_interactions)
print(f"Digital roots of triplets: {triplet_roots}")

def generate_energy_nodes(triplet_roots, time_steps, iteration=1):
    """
    Generates energy nodes based on digital root triplet patterns and calculates
    their relativistic energy interactions in quantum vacuum, with iterative strengthening.
    """
    # Physical constants
    c = 2.998e8  # Speed of light in m/s
    h = 6.626e-34  # Planck constant

    # Strengthening factor increases with each iteration
    strength_multiplier = (1 + golden_ratio) ** (iteration - 1)

    # Node generation parameters
    num_nodes = len(triplet_roots)
    node_spacing = golden_ratio * strength_multiplier

    # Initialize node arrays
    nodes = np.zeros((num_nodes, 3))
    node_energies = np.zeros((num_nodes, time_steps))
    node_interactions = np.zeros((num_nodes, num_nodes, time_steps))

    # Generate strengthened node positions
    for i, root in enumerate(triplet_roots):
        phi = 2 * np.pi * root / 9
        theta = np.pi * fibonacci_sequence(i+1)[-1] / 3

        nodes[i] = [
            node_spacing * i * np.sin(theta) * np.cos(phi),
            node_spacing * i * np.sin(theta) * np.sin(phi),
            node_spacing * i * np.cos(theta)
        ]

    accumulated_energy = 0

    # Calculate enhanced node energies and interactions
    for t in range(time_steps):
        for i in range(num_nodes):
            # Enhanced base energy
            base_energy = triplet_roots[i] * h * (1 + np.sin(t/10)) * strength_multiplier

            # Enhanced quantum tunneling
            tunneling = sum([
                0.1 * triplet_roots[j] * h * np.exp(-np.linalg.norm(nodes[i]-nodes[j])) * strength_multiplier
                for j in range(num_nodes) if j != i
            ])

            # Enhanced relativistic energy calculation
            rest_energy = base_energy + tunneling
            velocity = 0.1 * c * (1 + np.sin(t/20)) * strength_multiplier
            velocity = min(velocity, 0.99 * c)  # Ensure we don't exceed speed of light
            gamma = 1 / np.sqrt(1 - (velocity ** 2) / (c ** 2))
            node_energies[i,t] = gamma * rest_energy

            # Enhanced node interactions
            for j in range(i+1, num_nodes):
                distance = max(np.linalg.norm(nodes[i]-nodes[j]), 1e-10)  # Prevent division by zero
                interaction_strength = (node_energies[i,t] * node_energies[j,t]) / (distance ** 2)
                node_interactions[i,j,t] = interaction_strength * strength_multiplier
                node_interactions[j,i,t] = interaction_strength * strength_multiplier

            accumulated_energy += node_energies[i,t]

    return nodes, node_energies, node_interactions, accumulated_energy

def iterative_energy_enhancement(triplet_roots, time_steps, max_iterations=5, energy_threshold=1e6):
    """
    Repeatedly generate and enhance energy nodes, feeding energy back into reservoir
    """
    total_reservoir_energy = initial_reservoir
    iteration_results = []

    for iteration in range(1, max_iterations + 1):
        # Generate nodes with current strength
        nodes, node_energies, node_interactions, new_energy = generate_energy_nodes(
            triplet_roots, time_steps, iteration)

        # Add new energy to reservoir
        total_reservoir_energy += new_energy

        # Store results
        iteration_results.append({
            'iteration': iteration,
            'nodes': nodes,
            'energies': node_energies,
            'interactions': node_interactions,
            'new_energy': new_energy,
            'total_reservoir': total_reservoir_energy
        })

        print(f"\nIteration {iteration} Results:")
        print(f"New Energy Generated: {new_energy:.2e} J")
        print(f"Total Reservoir Energy: {total_reservoir_energy:.2e} J")

        # Check if energy threshold reached
        if total_reservoir_energy > energy_threshold:
            print(f"\nEnergy threshold {energy_threshold:.2e} J reached after {iteration} iterations")
            break

    return iteration_results, total_reservoir_energy

# Run iterative enhancement
iteration_results, final_reservoir_energy = iterative_energy_enhancement(
    triplet_roots, num_steps, max_iterations=5, energy_threshold=1e6)

# Visualize final enhanced state
plt.figure(figsize=(15, 10))

final_result = iteration_results[-1]

# Plot final node positions in 3D
ax1 = plt.subplot(2, 2, 1, projection='3d')
ax1.scatter(final_result['nodes'][:,0],
           final_result['nodes'][:,1],
           final_result['nodes'][:,2],
           c=triplet_roots, cmap='viridis', s=100)
ax1.set_title(f'Enhanced Energy Nodes (Iteration {len(iteration_results)})')

# Plot energy evolution across iterations
plt.subplot(2, 2, 2)
iteration_energies = [result['new_energy'] for result in iteration_results]
plt.plot(range(1, len(iteration_results) + 1), iteration_energies)
plt.xlabel('Iteration')
plt.ylabel('Energy Generated (J)')
plt.title('Energy Generation Progress')

# Plot final node energies
plt.subplot(2, 2, 3)
for i in range(len(final_result['nodes'])):
    plt.plot(final_result['energies'][i], label=f'Node {i}')
plt.xlabel('Time Steps')
plt.ylabel('Energy (J)')
plt.title('Final Node Energy States')

# Plot reservoir growth
plt.subplot(2, 2, 4)
reservoir_levels = [result['total_reservoir'] for result in iteration_results]
plt.plot(range(1, len(iteration_results) + 1), reservoir_levels)
plt.xlabel('Iteration')
plt.ylabel('Total Reservoir Energy (J)')
plt.title('Energy Reservoir Growth')

plt.tight_layout()
plt.show()

print("\nFinal System Analysis:")
print(f"Initial Reservoir Energy: {initial_reservoir:.2e} J")
print(f"Final Reservoir Energy: {final_reservoir_energy:.2e} J")
print(f"Energy Gain Factor: {final_reservoir_energy/initial_reservoir:.2f}x")
print(f"Number of Iterations: {len(iteration_results)}")
