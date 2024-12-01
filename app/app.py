import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from Env import CabDriver  # Ensure Env.py is accessible

# Initialize the environment
env = CabDriver()

# Define the DQN model structure
def build_model():
    model = Sequential()
    model.add(Dense(units=200, input_dim=env.state_size_arch_2, activation='relu', name="DHLayer-1"))
    model.add(Dense(units=150, activation='relu', name="DHLayer-2"))
    model.add(Dense(units=100, activation='relu', name="DHLayer-3"))
    model.add(Dense(units=env.action_size, activation='linear', name="Output"))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Instantiate and load model weights
model = build_model()
model.load_weights("dqn_agent_weights_final_arch2.weights.h5")

# Function to predict the best request based on multiple pickup-drop pairs and the driver's current location
def get_best_request(current_location, pickup_list, drop_list, time, day):
    best_request = None
    best_q_value = -float('inf')  # Initialize with a very low value to find max Q-value
    all_q_values = []  # List to store Q-values for each request
    
    # Loop through each pickup-drop pair in the request list
    for pickup, drop in zip(pickup_list, drop_list):
        # Create the state based on the current location, time, and day
        state = (current_location, time, day)
        state_encd = env.state_encod_arch_2(state)  # Encode state for model input
        state_encd = np.reshape(state_encd, [1, model.input_shape[1]])  # Reshape for model compatibility
        
        # Predict Q-values for all possible actions in this state
        q_values = model.predict(state_encd, verbose=0)
        
        # Find the Q-value for the specific action (pickup, drop)
        action_index = env.action_space.index((pickup, drop))  # Find the index of (pickup, drop) action
        q_value = q_values[0][action_index]  # Get the Q-value for this action
        
        # Store the Q-value and action in the list
        all_q_values.append(((pickup, drop), q_value))
        
        # Check if this is the best Q-value so far
        if q_value > best_q_value:
            best_q_value = q_value
            best_request = (pickup, drop)  # Update the best request
    
    return best_request, best_q_value, all_q_values  # Return the best request, its Q-value, and all Q-values

# Define the number of cities
m = 5  # Number of cities, ranges from 0 to m-1

# Streamlit interface setup
st.title("Cab Driver Decision Assistant")
st.markdown("Enter multiple pickup and drop-off requests and let the model suggest the best request.")

# User input for current location, time, and day
current_location = st.number_input("Enter Driver's Current Location (ID)", min_value=0, max_value=m - 1)

# User input for pickup-drop pairs using multiselect
pairs = [(p, d) for p in range(m) for d in range(m) if p != d]  # Generate all valid pairs
selected_pairs = st.multiselect("Select Pickup and Drop-off Pairs", options=pairs, format_func=lambda x: f"Pickup {x[0]} -> Drop {x[1]}")

# Convert selected pairs into pickup and drop-off lists
pickup_list = [pair[0] for pair in selected_pairs]
drop_list = [pair[1] for pair in selected_pairs]

time = st.slider("Enter Time of Day (0-23)", min_value=0, max_value=23)
day = st.selectbox("Enter Day of the Week", options=list(range(7)), format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

if st.button("Get Best Request"):
    if selected_pairs:
        # Predict the best request from the given pairs
        best_request, best_q_value, all_q_values = get_best_request(current_location, pickup_list, drop_list, time, day)
        
        # Display the result
        if best_request:
            st.subheader("Best Request to Accept:")
            st.write(f"Pickup {best_request[0]} -> Drop {best_request[1]}")
            st.write(f"Expected Reward: {best_q_value:.2f}")
            
            # Display all Q-values
            st.markdown("### Q-Values for All Requests")
            for (pickup, drop), q_value in all_q_values:
                st.write(f"Request (Pickup {pickup} -> Drop {drop}): Q-value = {q_value:.2f}")
        else:
            st.write("No valid request found.")
    else:
        st.write("Please select at least one pickup-drop-off pair.")