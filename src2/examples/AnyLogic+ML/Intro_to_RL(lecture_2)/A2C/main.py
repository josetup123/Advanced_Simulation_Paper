"""
IE-608 DQN example.
"""

# import
import logging
import numpy as np
from DRL import Agent, DRL_Env
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.orm import sessionmaker
import os
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)



def create_left_upper_triangular_matrix(num_rows, decimals=3, seed=42):
    # Set the seed for reproducibility
    np.random.seed(seed)

    # Initialize an empty matrix with the desired shape
    matrix = np.zeros((num_rows, num_rows), dtype=float)

    # Populate the matrix such that each row is left-upper triangular and sums to 1
    for i in range(num_rows):
        # Generate random probabilities for the upper triangular part of the row
        row_probabilities = np.random.dirichlet(np.ones(num_rows - i))
        # Pad the remaining elements in the row with zeros
        row = np.pad(row_probabilities, (i, 0), mode='constant')
        # Round the probabilities to the specified decimals
        row = np.round(row, decimals=decimals)
        # Assign the row to the matrix
        matrix[i, :] = row

    return np.mat(matrix)



def main():
    """
    mian
    """




    parser = argparse.ArgumentParser()

    parser.add_argument("-Max_FirePoints",
                        "--Max_FirePoints",
                        help="Max_FirePoints",
                        type=int)
    args = parser.parse_args()
    Max_FirePoints = int(args.Max_FirePoints)


    


    # --------- parameters -----------

    EPISODES=2000 #(FROM ANULOGIC)
    DISCOUNT_FACTOR=1.0 #(FROM ANULOGIC)

    seed_value = 42 #(FROM ANYLIGIUC)

    # number of states
    num_states = Max_FirePoints # (FROM ANYLOGIC) THESE STATES WILL BE DIFFERENT DECISIONS OR FIRESPOTS THE FITHFIGHTER WILL HAVE TO DECIDE WHETHER TO DO CERTAIN ACTION!

    # number of plans
    num_plans = 2  #FROM ANYLOGIC#FUNCTIONAL OF NEED REPAIR (PUT OFF THE FIRE OR NOT PUTTING IT OFF!!!!)

    # horizon
    T = 20   #HOW MANY TIME STEPS BEFORE I UPDATE THE POLICY OR FOR HOW MUCH TIME I WILL SIMULATE LETS SET 20 HOURS!

    # success probability
    success_pr = [0.0, 0.5, 0.9] #PROBABILITY OF PUTTING OFF THE FGIRE!

    # degrade pr, implies four states
    degrade_pr = create_left_upper_triangular_matrix(num_states, seed=seed_value)

    print(degrade_pr)
    
    # degrade_pr_2=np.mat([
    #     [0.3, 0.5, 0.2, 0.0], #PROBABILITY OF EACH STATE (0,1,2,3) HAVE TO SUM 1
    #     [0.0, 0.4, 0.4, 0.2], #EACH RPW REPRESENTS A STATE, EACH COLUMN IS A STATE AS WELL!
    #     [0.0, 0.0, 0.4, 0.6],
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    # print(degrade_pr_2)

    # revenue
    C_r = 10 #THIS IS THE REWARD OF HAVING NO WILDFIRES OVER OVER TIME PER EACH TIME STEP!!!

    # operating cost --> FIRE EFFECTS COST WHEN THE FIRE STATE IS HIGHER THEN THE COST WILL BE HIGHER!!!!!!
    C_o = [
        i for i in range(num_states) #THESE COULD BE DYNAMIC, 
    ]
    print(C_o)
    # maintenance cost --> THIS IS THE COST OF PUTTING OFF A FIRE!!! EFFECT OVER POPULATION OVER AREA! OR ON THE BAISIS OF AREA DENSITY!!
    C_m = [
        i for i in range(num_plans) #THESE COULD BE DYNAMIC
    ]
    print(C_m)

    # ------------ MDP --------------

    # initial state: t, condition
    initial_state = (0, 0)

    # number of actions
    actions = range(num_plans)

    # transition function
    def trans_func(t, s, a):
        """
        transition
        """
        # terminal
        if t >= T:
            return "Delta"
        # calculate pr
        pr = []
        for i in range(num_states):
            # smaller state
            if i < s[1]:
                pr.append(
                    success_pr[a] * degrade_pr[0, i]
                )
            else:
                pr.append(np.sum([
                    success_pr[a] * degrade_pr[0, i],
                    (1 - success_pr[a]) * degrade_pr[s[1], i]
                ]))
        # sample new state according to pr JT WORK ON IT!
        
        return (t + 1, np.random.choice(
            range(num_states), size=1, replace=False,
            p=pr
        )[0])
    
    # reward function
    def reward_func(s, a, s_new):
        """
        reward
        """
        return C_r - C_o[s[1]] - C_m[a]

    # agent
    manager = Agent(
        name="manager",
        actions=actions,
        input_size=2,
        actor_hidden_layers=[100],
        critic_hidden_layers=[100],
        output_size=len(actions),
        actor_lr=1e-5,
        critic_lr=1e-5
    )
    

    # define problem
    problem = DRL_Env(
        name="maintenance",
        agent=manager,
        initial_state=initial_state,
        trans_func=trans_func,
        reward_func=reward_func
    )

    # logging
    logging.basicConfig(
        filename='maintenance.log', filemode='w+',
        format='%(levelname)s - %(message)s', level=logging.INFO
    )

    # DQN
    G = problem.advantage_actor_acritic(
        episodes=EPISODES, #JT CJAGNERS! 5000
        discount_factor=DISCOUNT_FACTOR,
        write_log=True
    )

    # plot
    manager.plot_loss(dir='figs/')

    manager.plot_G(dir='figs/')

    #HERE WE PARSE THE LOG PUSH IT TO DATABASE!!! THE INFO WE NEED!!!!
    # Specify the total number of lines in the file
    # Specify the path to the log file
    log_file_path = dir_path + "/" + 'maintenance.log'

    # Calculate the total number of lines in the log file
    total_lines = sum(1 for line in open(log_file_path))

    # Number of lines to skip (total_lines - 86)
    skip_rows = total_lines - 86

    # Read the last 86 lines using skiprows
    parse = pd.read_csv(log_file_path, skiprows=range(1, skip_rows), header=None, delimiter='\t')
    output_data=list(parse[0].values)

    # Display the parsed data
    print(parse)


    rows = []
    current_epoch = None
    current_row = {"epochs": None, "state": None, "action": None, "reward": None, "return": None}

    # Process each line
    for line in output_data:
        # Split the line into words
        words = line.split()

        # Check if the line contains "epoch:"
        if "epoch:" in words:
            # If a new epoch starts, save the current row and initialize a new one
            if current_row["epochs"] is not None:
                rows.append(current_row)
            current_epoch = int(words[-1])
            current_row = {"epochs": current_epoch, "state": None, "action": None, "reward": None, "return": None}
        elif "state:" in words:
            current_row["state"] = " ".join(words[2:])
        elif "action:" in words:
            current_row["action"] = int(words[-1])
        elif "reward:" in words:
            current_row["reward"] = int(words[-1])
        elif "return:" in words:
            current_row["return"] = int(words[-1])

    # Append the last row
    if current_row["epochs"] is not None:
        rows.append(current_row)

    # Create a DataFrame from the processed rows
    df = pd.DataFrame(rows)

    # Set the order of columns
    df = df[['epochs', 'reward', 'state', 'action', 'return']]

    # Display the DataFrame
    print(df)

    # Extract the 'return' value from the last element of output_data
    # last_return_line = [line for line in output_data if "return:" in line][-1]
    # last_return = int(last_return_line.split()[-1]) if last_return_line else None
    # print(f"Return Value from Last Line: {last_return}")





    return 0 


if __name__ == "__main__":
    main()
