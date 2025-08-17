# app.py

import streamlit as st
import numpy as np
import pandas as pd
from tqdm import trange # Used for progress bar in simulation
from bandit_model import Bandit # Import the Bandit class

# -----------------------------------------------------------------------------
# ---- Streamlit Application Setup ----
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Multi-Armed Bandit Algorithms", layout="centered")
st.title("🤖 Multi-Armed Bandit Algorithms : Amazon Add Titles for same product")
st.markdown("""
This application demonstrates different bandit strategies for choosing ads:
- **ε-Greedy**: Balances **exploration** (trying new ads) and **exploitation** (picking the best known ad).
- **UCB (Upper Confidence Bound)**: Optimistically explores ads with higher uncertainty, ensuring all promising options are eventually tried.
""")

# ---- Session State Initialization ----
# All critical variables are stored in Streamlit's session state
# so they persist across reruns (e.g., button clicks).
if "bandit" not in st.session_state:
    st.session_state.bandit = Bandit(k_arm=3, epsilon=0.1, sample_averages=True) # Default to epsilon-greedy
    st.session_state.bandit.reset() # Initialize the bandit for the first time
    st.session_state.total_reward = 0
    st.session_state.history = [] # Stores (ad_index, reward_binary_0_or_1) for plotting and table
    st.session_state.current_ad = st.session_state.bandit.act() # Get the very first ad
    st.session_state.previous_ad = None # To display what the previous ad was
    st.session_state.initial_q_true = st.session_state.bandit.q_true.copy() # Store true values for consistent simulations

# Define the names of the ads
ads = ["Shoes 👟 at full disount of 50%", "Branded shoes at half prices for smart customer 👟", "Pizza free with Branded shoes 👟 + 🍕"]

# -----------------------------------------------------------------------------
# ---- Sidebar for Algorithm Selection ----
# -----------------------------------------------------------------------------
st.sidebar.header("Algorithm Settings")
algorithm = st.sidebar.radio("Select Algorithm", ["ε-Greedy", "UCB"], key="algo_select")

# Sliders for algorithm-specific parameters (disabled if not relevant to selected algorithm)
epsilon_param = st.sidebar.slider("ε (epsilon) for ε-Greedy", 0.0, 1.0, 0.1, 0.05, key="epsilon_slider",
                                   disabled=(algorithm != "ε-Greedy"))
ucb_c_param = st.sidebar.slider("UCB Parameter (c) for UCB", 0.0, 5.0, 2.0, 0.1, key="ucb_slider",
                                  disabled=(algorithm != "UCB"))

if st.sidebar.button("Apply Settings & Reset", key="apply_reset_button"):
    # Clear history and reset rewards for a fresh start with new settings
    st.session_state.history = []
    st.session_state.total_reward = 0
    st.session_state.previous_ad = None
    
    # Initialize the bandit based on selected algorithm and parameters
    if algorithm == "ε-Greedy":
        st.session_state.bandit = Bandit(k_arm=3, epsilon=epsilon_param, sample_averages=True)
    elif algorithm == "UCB":
        st.session_state.bandit = Bandit(k_arm=3, UCB_param=ucb_c_param, sample_averages=True)
    
    st.session_state.bandit.reset() # Reset the bandit's internal state
    st.session_state.current_ad = st.session_state.bandit.act() # Get the first ad for the new setup
    st.session_state.initial_q_true = st.session_state.bandit.q_true.copy() # Update true values for new simulations
    st.rerun() # Rerun the app to apply changes and update UI

st.sidebar.markdown("---")
st.sidebar.info("Adjust parameters and click 'Apply Settings & Reset' to reconfigure the bandit algorithm.")


## ---- Interactive Mode ----
st.subheader("👨‍💻 Interactive Mode")
st.markdown("Interact with the bandit by clicking or skipping ads. The bandit learns from your actions!")

# Store the current ad index before a new one is selected on user interaction
st.session_state.previous_ad = st.session_state.current_ad

# Display the current ad
i = st.session_state.current_ad
st.markdown(f"## 📢 Current Ad: **{ads[i]}**")

# Display the previous ad if an action has occurred and it's different
if st.session_state.previous_ad is not None and st.session_state.previous_ad != st.session_state.current_ad:
    st.markdown(f"**Previous Ad:** {ads[st.session_state.previous_ad]}")

col1, col2 = st.columns(2)
with col1:
    if st.button("✅ Click", key="click_button"):
        # When user clicks, explicitly pass a reward of 1 to the bandit for learning
        st.session_state.bandit.step(i, external_reward=1) 
        
        # Log 1 for a click in history for consistent plotting and table
        st.session_state.history.append((i, 1))
        st.session_state.total_reward += 1
        st.success(f"You clicked: {ads[i]}")
        
        # Get the next ad from the bandit based on updated estimates
        st.session_state.current_ad = st.session_state.bandit.act()
        st.rerun() # Force rerun to update the UI with the next ad

with col2:
    if st.button("❌ Skip", key="skip_button"):
        # When user skips, explicitly pass a reward of 0 to the bandit for learning
        st.session_state.bandit.step(i, external_reward=0)
        
        # Log 0 for a skip in history (no click)
        st.session_state.history.append((i, 0))
        st.info(f"You skipped: {ads[i]}")
        
        # Get the next ad from the bandit based on updated estimates
        st.session_state.current_ad = st.session_state.bandit.act()
        st.rerun() # Force rerun to update the UI with the next ad

st.markdown("---")

## ---- Auto Simulation Mode ----
st.subheader("⚡ Auto Simulation Mode")
st.markdown("Run the selected bandit algorithm automatically for a specified number of rounds.")

n_rounds = st.slider("Number of simulation rounds", 10, 1000, 100, 10, key="n_rounds_slider")

if st.button("▶ Run Auto Simulation", key="run_simulation_button"):
    st.session_state.previous_ad = None # Clear previous ad display for simulation context
    
    bandit = st.session_state.bandit
    # Reset bandit for a new simulation run, ensuring it starts from a fresh state
    # Use the initially stored q_true values for consistent true arm rewards across simulations
    bandit.reset()
    bandit.q_true = st.session_state.initial_q_true.copy() 
    
    # Clear the existing history and total reward before starting a new simulation
    st.session_state.history = []
    st.session_state.total_reward = 0
    
    # Run the simulation for n_rounds
    for _ in trange(n_rounds, desc="Simulating"):
        action = bandit.act() # Select an action
        # For simulation, we generate the reward from the ad's true distribution (Gaussian)
        reward_value = np.random.randn() + bandit.q_true[action]
        bandit.step(action, external_reward=reward_value) # Pass the generated reward to the step method
        
        # Add to history: (action_index, binary_reward_for_plotting)
        # We convert the continuous reward to binary (1 for positive reward, 0 otherwise) for 'clicks'
        is_click = 1 if reward_value > 0 else 0 
        st.session_state.history.append((action, is_click))
        st.session_state.total_reward += is_click # Summing binary clicks
        
    st.success(f"Simulation completed for {n_rounds} rounds!")
    st.session_state.current_ad = st.session_state.bandit.act() # Get a new ad after simulation
    st.rerun() # Rerun to update results and current ad display

st.markdown("---")

## ---- Results Section ----
st.subheader("📊 Results So Far")
st.markdown("Review the performance of the bandit algorithm, including ad counts, estimated values, and cumulative rewards.")

if st.session_state.history:
    counts = st.session_state.bandit.action_count.astype(int)
    values = st.session_state.bandit.q_estimation
    true_q_values = st.session_state.bandit.q_true # Use q_true from the bandit

    # Create a history DataFrame to easily count clicks for each ad
    history_df = pd.DataFrame(st.session_state.history, columns=["Ad", "Reward"])
    # Group by Ad index and sum the 'Reward' (binary 0/1) to get total clicks per ad
    # .reindex ensures all ads are included, even if not shown/clicked yet
    clicks_by_ad = history_df.groupby("Ad")["Reward"].sum().reindex(range(len(ads)), fill_value=0).astype(int)
    
    results_data = []
    for idx, ad_text in enumerate(ads):
        results_data.append({
            "Ad": ad_text,
            "Times Shown": counts[idx],
            "Times Clicked": clicks_by_ad.loc[idx], # Show how many times each ad was clicked
            "Estimated Value": round(values[idx], 3), # Estimated mean reward by the bandit
            "True Value (Mean Reward)": round(true_q_values[idx], 3) # The actual mean reward for this ad
        })

    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results)

    st.write("**Total Reward (Cumulative Clicks):**", st.session_state.total_reward)

    st.markdown("#### Cumulative Reward Over Time")
    st.line_chart(history_df["Reward"].cumsum(), use_container_width=True)
    st.markdown("*(X-axis: Actions Shown/Time Steps, Y-axis: Cumulative Clicks/Reward)*")

    # Best Ad Recommendation based on current estimations
    best_ad_index = np.argmax(st.session_state.bandit.q_estimation)
    st.markdown(f"### 🏆 Best Ad Recommendation (so far): **{ads[best_ad_index]}**")