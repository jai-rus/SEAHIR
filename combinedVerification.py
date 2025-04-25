import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# 1. Load and prepare Orange County real-world data
# ------------------------------
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

# Normalize cumulative cases to model population
orange_population = 3_200_000
model_population = 1000
date_filtered['cumulative_normalized'] = cumulative_cases / orange_population * model_population

# Get normalized cumulative cases for plotting
real_cumulative = date_filtered['cumulative_normalized'].values
days = np.arange(len(real_cumulative))

#FOR SEAHIR
'''def extract_cumulative_infected(raw_data):
    """
    Extracts cumulative infected from SEAHIR simulation.
    Assumes keys are time steps, and values are dicts of compartment counts.
    """
    time_steps = sorted(raw_data.keys())
    cumulative = []
    for t in time_steps:
        infected_total = sum(raw_data[t][i] for i in [1, 2, 3, 4, 5])  # E + A + H + I + R
        cumulative.append(infected_total)
    return time_steps, cumulative'''
def extract_cumulative_infected(raw_data):
    """
    Extracts cumulative infected from SEIR simulation.
    Assumes keys are time steps, and values are dicts of compartment counts.
    """
    time_steps = sorted(raw_data.keys())
    cumulative = []
    for t in time_steps:
        infected_total = raw_data[t][2] + raw_data[t][3]  # I + R
        cumulative.append(infected_total)
    return time_steps, cumulative


# Replace this with your actual raw_data
raw_data = {
1: {0: 931, 1: 69, 2: 0, 3: 0},
2: {0: 867, 1: 133, 2: 0, 3: 0},
3: {0: 821, 1: 179, 2: 0, 3: 0},
4: {0: 761, 1: 239, 2: 0, 3: 0},
5: {0: 709, 1: 291, 2: 0, 3: 0},
6: {0: 669, 1: 262, 2: 69, 3: 0},
7: {0: 647, 1: 220, 2: 133, 3: 0},
8: {0: 632, 1: 189, 2: 179, 3: 0},
9: {0: 622, 1: 139, 2: 239, 3: 0},
10: {0: 614, 1: 95, 2: 291, 3: 0},
11: {0: 608, 1: 61, 2: 331, 3: 0},
12: {0: 599, 1: 48, 2: 353, 3: 0},
13: {0: 592, 1: 40, 2: 368, 3: 0},
14: {0: 585, 1: 37, 2: 378, 3: 0},
15: {0: 578, 1: 36, 2: 386, 3: 0},
16: {0: 570, 1: 38, 2: 323, 3: 69},
17: {0: 564, 1: 35, 2: 268, 3: 133},
18: {0: 553, 1: 39, 2: 229, 3: 179},
19: {0: 528, 1: 57, 2: 176, 3: 239},
20: {0: 513, 1: 65, 2: 131, 3: 291},
21: {0: 496, 1: 74, 2: 99, 3: 331},
22: {0: 471, 1: 93, 2: 83, 3: 353},
23: {0: 444, 1: 109, 2: 79, 3: 368},
24: {0: 418, 1: 110, 2: 94, 3: 378},
25: {0: 399, 1: 114, 2: 101, 3: 386},
26: {0: 389, 1: 107, 2: 112, 3: 392},
27: {0: 373, 1: 98, 2: 128, 3: 401},
28: {0: 354, 1: 90, 2: 148, 3: 408},
29: {0: 343, 1: 75, 2: 167, 3: 415},
30: {0: 333, 1: 66, 2: 179, 3: 422},
31: {0: 313, 1: 76, 2: 181, 3: 430},
32: {0: 301, 1: 72, 2: 191, 3: 436},
33: {0: 291, 1: 63, 2: 199, 3: 447},
34: {0: 281, 1: 62, 2: 185, 3: 472},
35: {0: 272, 1: 61, 2: 180, 3: 487},
36: {0: 266, 1: 47, 2: 183, 3: 504},
37: {0: 252, 1: 49, 2: 170, 3: 529},
38: {0: 247, 1: 44, 2: 153, 3: 556},
39: {0: 238, 1: 43, 2: 137, 3: 582},
40: {0: 232, 1: 40, 2: 127, 3: 601},
41: {0: 223, 1: 43, 2: 123, 3: 611},
42: {0: 220, 1: 32, 2: 121, 3: 627},
43: {0: 217, 1: 30, 2: 107, 3: 646},
44: {0: 213, 1: 25, 2: 105, 3: 657},
45: {0: 208, 1: 24, 2: 101, 3: 667},
46: {0: 198, 1: 25, 2: 90, 3: 687},
47: {0: 193, 1: 27, 2: 81, 3: 699},
48: {0: 185, 1: 32, 2: 74, 3: 709},
49: {0: 176, 1: 37, 2: 68, 3: 719},
50: {0: 169, 1: 39, 2: 64, 3: 728},
51: {0: 161, 1: 37, 2: 68, 3: 734},
52: {0: 155, 1: 38, 2: 59, 3: 748},
53: {0: 149, 1: 36, 2: 62, 3: 753},
54: {0: 144, 1: 32, 2: 62, 3: 762},
55: {0: 138, 1: 31, 2: 63, 3: 768},
56: {0: 132, 1: 29, 2: 62, 3: 777},
57: {0: 127, 1: 28, 2: 65, 3: 780},
58: {0: 125, 1: 24, 2: 68, 3: 783},
59: {0: 123, 1: 21, 2: 69, 3: 787},
60: {0: 121, 1: 17, 2: 70, 3: 792},
61: {0: 118, 1: 14, 2: 66, 3: 802},
62: {0: 113, 1: 14, 2: 66, 3: 807},
63: {0: 108, 1: 17, 2: 60, 3: 815},
64: {0: 101, 1: 22, 2: 53, 3: 824},
65: {0: 99, 1: 22, 2: 48, 3: 831},
66: {0: 98, 1: 20, 2: 43, 3: 839},
67: {0: 96, 1: 17, 2: 42, 3: 845},
68: {0: 94, 1: 14, 2: 41, 3: 851},
69: {0: 92, 1: 9, 2: 43, 3: 856},
70: {0: 89, 1: 10, 2: 39, 3: 862},
71: {0: 89, 1: 9, 2: 34, 3: 868},
72: {0: 85, 1: 11, 2: 31, 3: 873},
73: {0: 82, 1: 12, 2: 31, 3: 875},
74: {0: 82, 1: 10, 2: 31, 3: 877},
75: {0: 82, 1: 7, 2: 32, 3: 879},
76: {0: 79, 1: 10, 2: 29, 3: 882},
77: {0: 77, 1: 8, 2: 28, 3: 887},
78: {0: 74, 1: 8, 2: 26, 3: 892},
79: {0: 70, 1: 12, 2: 19, 3: 899},
80: {0: 66, 1: 16, 2: 17, 3: 901},
81: {0: 64, 1: 15, 2: 19, 3: 902},
82: {0: 61, 1: 16, 2: 19, 3: 904},
83: {0: 61, 1: 13, 2: 20, 3: 906},
84: {0: 60, 1: 10, 2: 22, 3: 908},
85: {0: 58, 1: 8, 2: 23, 3: 911},
86: {0: 56, 1: 8, 2: 25, 3: 911},
87: {0: 55, 1: 6, 2: 24, 3: 915},
88: {0: 54, 1: 7, 2: 21, 3: 918},
89: {0: 51, 1: 9, 2: 22, 3: 918},
90: {0: 47, 1: 11, 2: 24, 3: 918},
91: {0: 44, 1: 12, 2: 23, 3: 921},
92: {0: 43, 1: 12, 2: 22, 3: 923},
93: {0: 41, 1: 13, 2: 20, 3: 926},
94: {0: 41, 1: 10, 2: 19, 3: 930},
95: {0: 39, 1: 8, 2: 19, 3: 934},
96: {0: 39, 1: 5, 2: 20, 3: 936},
97: {0: 39, 1: 4, 2: 18, 3: 939},
98: {0: 39, 1: 2, 2: 20, 3: 939},
99: {0: 39, 1: 2, 2: 19, 3: 940},
100: {0: 38, 1: 1, 2: 19, 3: 942},
}
simulated_days, cumulative_infected = extract_cumulative_infected(raw_data)

# ------------------------------
# 3. Plot cumulative cases comparison
# ------------------------------
# Use raw cumulative cases (skip normalization)
real_cumulative = date_filtered['cases'].values  
days = np.arange(len(real_cumulative))

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the SEAHIR model on the primary y-axis (linear)
ax1.plot(
    simulated_days[:len(days)], 
    cumulative_infected[:len(days)], 
    label="SEAHIR Model (Cumulative)", 
    linestyle='--', 
    color='black'
)
ax1.set_xlabel("Days since 2020-03-01")
ax1.set_ylabel("Model Cumulative (Linear Scale)", color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Create a secondary y-axis (log) for real data
ax2 = ax1.twinx()
ax2.plot(
    days, 
    real_cumulative, 
    label="Orange County Real Data (Cumulative)", 
    linestyle='-', 
    color='orange',
    marker='o',
    markersize=4
)
ax2.set_yscale('log')
ax2.set_ylabel("Real Data (Log Scale)", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Title and layout
fig.suptitle("Cumulative COVID-19 Cases: Model vs Real Data (March-April 2020)")
fig.tight_layout()
plt.grid(True)

plt.savefig('/home/jmlegion/repast/SEAHIR/validation.png')
