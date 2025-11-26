import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#load dataset
df = pd.read_csv('/home/jmlegion/repast/SEAHIR/archive/us_counties_covid19_daily.csv')

# Filter for Orange County, California
'''orange = df[
    (df['state'].str.lower() == 'california') & 
    (df['county'].str.lower() == 'orange')
].copy()'''
validation = df[
    (df['state'].str.lower() == 'california') & 
    (df['county'].str.lower() == 'los angeles')
].copy()
# Convert date and sort
validation['date'] = pd.to_datetime(validation['date'])
validation = validation.sort_values('date').reset_index(drop=True)

# Define date range
start_date = pd.to_datetime('2020-03-01')
end_date = pd.to_datetime('2020-06-01')
date_filtered = validation[(validation['date'] >= start_date) & (validation['date'] <= end_date)].copy()

# Check if 'cases' is cumulative (monotonically increasing)
if date_filtered['cases'].is_monotonic_increasing:
    print("'cases' column is cumulative - using directly")
    cumulative_cases = date_filtered['cases']
else:
    print("'cases' column is daily new cases - computing cumulative sum")
    cumulative_cases = date_filtered['cases'].cumsum()

# Normalize to model population (1000)
# Using actual population for normalization
la_population = 10_000_000
model_population = 1000
date_filtered['cumulative_normalized'] = cumulative_cases / la_population * model_population

# Get normalized cumulative cases for plotting
real_cumulative = date_filtered['cumulative_normalized'].values
days = np.arange(len(real_cumulative))

#FOR SEAHIR
def extract_cumulative_infected(raw_data):
    """
    Extracts cumulative infected from SEAHIR simulation.
    Assumes keys are time steps, and values are dicts of compartment counts.
    """
    time_steps = sorted(raw_data.keys())
    cumulative = []
    for t in time_steps:
        infected_total = sum(raw_data[t][i] for i in [1, 2, 3, 4, 5])  # E + A + H + I + R
        cumulative.append(infected_total)
    return time_steps, cumulative

#FOR SEIR
'''def extract_cumulative_infected(raw_data):
    """
    Extracts cumulative infected from SEIR simulation.
    Assumes keys are time steps, and values are dicts of compartment counts.
    """
    time_steps = sorted(raw_data.keys())
    cumulative = []
    for t in time_steps:
        infected_total = raw_data[t][2] + raw_data[t][3]  # I + R
        cumulative.append(infected_total)
    return time_steps, cumulative'''


#Data from models
sim_df = pd.read_csv("/home/jmlegion/repast/SEAHIR/seahir_sim_data.csv")

# Extract cumulative infected = E + A + H + I + R
simulated_days = sim_df["day"].values
cumulative_infected = (
    sim_df["E"] + sim_df["A"] + sim_df["H"] + sim_df["I"] + sim_df["R"]
).values

# Both are now at same scale (model population of 1000)
model_cumulative = cumulative_infected
real_cumulative_scaled = real_cumulative

#plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Plot both on the same axis with same scale
ax.plot(
    simulated_days[:len(days)], 
    model_cumulative[:len(days)], 
    label="SEAHIR Model (Cumulative Infected)", 
    linestyle='--', 
    color='black',
    linewidth=2
)
ax.plot(
    days, 
    real_cumulative_scaled, 
    label="LA County Real Data (Normalized to Model Scale)", 
    linestyle='-', 
    color='orange',
    marker='o',
    markersize=4
)

ax.set_yscale('log')
ax.set_xlabel("Days since 2020-03-01", fontsize=12)
ax.set_ylabel("Cumulative Cases (Log Scale)", fontsize=12)
ax.set_title("Cumulative COVID-19 Cases: SEAHIR Model vs Real Data (March-June 2020)\nNormalized to Model Population (1000)", fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/jmlegion/repast/SEAHIR/validation_normalized.png', dpi=300)
plt.show()

print(f"\nModel population: {model_population}")
print(f"Real data population (LA County): ~{la_population:,}")
print(f"Normalization: Real cases scaled to model population of {model_population}")
print(f"\nDay 40 comparison:")
if len(days) > 39:
    print(f"  Model: {model_cumulative[39]:.0f} cumulative infected")
    print(f"  Real data (scaled): {real_cumulative_scaled[39]:.1f} cases")
print(f"\nFinal day comparison (day {len(days)}):")
print(f"  Model: {model_cumulative[len(days)-1]:.0f} cumulative infected")
print(f"  Real data (scaled): {real_cumulative_scaled[-1]:.1f} cases")