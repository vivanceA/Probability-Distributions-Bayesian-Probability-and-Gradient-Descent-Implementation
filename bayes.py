import random

def monty_hall_simulation(num_trials, switch=True):
    wins = 0

    for _ in range(num_trials):
        doors = ['goat', 'goat', 'car']  # Two goats, one car
        random.shuffle(doors)  # Randomly place the car

        # Player randomly picks a door
        player_choice = random.randint(0, 2)

        # Host opens a door with a goat (not the player's choice and not the car)
        remaining_doors = [i for i in range(3) if i != player_choice and doors[i] == 'goat']
        host_opens = random.choice(remaining_doors)

        # If switching, change choice to the other remaining door
        if switch:
            player_choice = [i for i in range(3) if i != player_choice and i != host_opens][0]

        # Check if the player won the car
        if doors[player_choice] == 'car':
            wins += 1

    win_percentage = (wins / num_trials) * 100
    strategy = "Switching" if switch else "Staying"
    print(f"{strategy} Strategy: Won {wins} times out of {num_trials} ({win_percentage:.2f}%)")

# Run the simulation for both strategies
monty_hall_simulation(num_trials=1000, switch=True)  # Switching strategy
monty_hall_simulation(num_trials=1000, switch=False) # Staying strategy
