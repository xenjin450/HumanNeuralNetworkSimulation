import pygame
import pygame_gui
import random
import math
import sys
from pygame.locals import QUIT, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, KEYDOWN, K_EQUALS, K_MINUS

# Initialize Pygame
pygame.init()

# Constants
WORLD_WIDTH, WORLD_HEIGHT = 1920, 1080
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FOOD_COLOR = (0, 255, 0)
WATER_COLOR = (0, 0, 255)
AGENT_BASE_COLOR = (200, 100, 100)
MUTATION_COLORS = {
    'A': (255, 0, 0),
    'B': (255, 165, 0),
    'C': (255, 255, 0),
    'D': (0, 128, 0),
    'E': (0, 255, 255),
    'F': (0, 0, 255),
    'G': (128, 0, 128),
    'H': (255, 192, 203),
    'I': (165, 42, 42),
    'J': (255, 215, 0),
    'K': (0, 255, 127),
    'L': (75, 0, 130),
    'M': (255, 20, 147),
    'N': (34, 139, 34),
    'O': (0, 191, 255),
    'P': (138, 43, 226),
    'Q': (255, 105, 180),
    'R': (64, 224, 208),
    'S': (173, 255, 47),
    'T': (255, 69, 0),
    'U': (210, 105, 30),
    'V': (154, 205, 50),
    'W': (0, 100, 0),
    'X': (255, 140, 0),
    'Y': (128, 0, 0),
    'Z': (0, 0, 128)
}

# Agent Settings
AGENT_SIZE = 12
CHILD_SIZE = 6
ADULT_AGE = 100.0  # Number of seconds to reach adult size
CONSUMPTION_RADIUS = 15
INITIAL_ENERGY = 100
BREED_ENERGY_THRESHOLD = 100
ENERGY_DECREMENT = 10  # Energy loss per second
MUTATION_PROBABILITY = 0.01  # 1% chance to mutate per update

# Resource Settings
INITIAL_FOOD = 50
INITIAL_WATER = 50
RESOURCE_REGENERATION_RATE = 1  # Probability per frame to regenerate resources

# Fonts
pygame.font.init()
FONT = pygame.font.SysFont('Arial', 14)

# Setup the display
screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
pygame.display.set_caption("Human-Based Simulation World")
clock = pygame.time.Clock()

# Setup GUI Manager
manager = pygame_gui.UIManager((WORLD_WIDTH, WORLD_HEIGHT))

# GUI Elements Original Properties
original_gui_elements = {}

# Function to create GUI elements and store their original properties
def create_gui_elements():
    global start_button, stop_button, restart_button, duration_label, duration_input
    global speed_up_button, slow_down_button, speed_label  # New GUI elements

    start_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((10, 10), (100, 40)),
        text='Start',
        manager=manager
    )
    stop_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((120, 10), (100, 40)),
        text='Stop',
        manager=manager
    )
    restart_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((230, 10), (100, 40)),
        text='Restart',
        manager=manager
    )
    duration_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((340, 10), (150, 40)),
        text='Duration (sec):',
        manager=manager
    )
    duration_input = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect((500, 10), (100, 40)),
        manager=manager
    )
    duration_input.set_text('60')  # Default duration

    # New GUI Elements for Speed Control
    speed_up_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((610, 10), (100, 40)),
        text='Speed Up',
        manager=manager
    )
    slow_down_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((720, 10), (100, 40)),
        text='Slow Down',
        manager=manager
    )
    speed_label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect((830, 10), (150, 40)),
        text='Speed: 1x',
        manager=manager
    )

    # Store original properties
    original_gui_elements['start_button'] = start_button.relative_rect.copy()
    original_gui_elements['stop_button'] = stop_button.relative_rect.copy()
    original_gui_elements['restart_button'] = restart_button.relative_rect.copy()
    original_gui_elements['duration_label'] = duration_label.relative_rect.copy()
    original_gui_elements['duration_input'] = duration_input.relative_rect.copy()
    original_gui_elements['speed_up_button'] = speed_up_button.relative_rect.copy()
    original_gui_elements['slow_down_button'] = slow_down_button.relative_rect.copy()
    original_gui_elements['speed_label'] = speed_label.relative_rect.copy()

# Initialize GUI Elements
create_gui_elements()

# Zoom and Pan Variables
zoom_factor = 1.0
offset_x, offset_y = 0, 0
dragging = False
last_mouse_pos = None

# Scaling Variables
gui_scale_factor = 1.0
MIN_GUI_SCALE = 0.5
MAX_GUI_SCALE = 2.0
SCALE_STEP = 0.1

# Simulation Control
running_simulation = False
simulation_duration = None  # In seconds

# Mutation Tracking
mutation_counts = {chr(i): 0 for i in range(65, 91)}  # 'A' to 'Z'

# Simulation Speed Control
time_scale = 1  # 1x speed
MAX_TIME_SCALE = 5
MIN_TIME_SCALE = 1

# Twin Probability
TWINS_PROBABILITY = 0.00001  # 0.001%

# Modified Advanced Neural Network Class with Backpropagation
class AdvancedNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        """
        Initialize the neural network.
        :param input_size: Number of input neurons.
        :param hidden_layers: List containing the number of neurons in each hidden layer.
        :param output_size: Number of output neurons.
        """
        self.layers = []

        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            # Weight matrix
            weight_matrix = [[random.uniform(-1, 1) for _ in range(layer_sizes[i + 1])] for _ in range(layer_sizes[i])]
            # Bias vector
            bias_vector = [random.uniform(-1, 1) for _ in range(layer_sizes[i + 1])]
            self.layers.append({
                'weights': weight_matrix,
                'biases': bias_vector,
                'outputs': [],
                'inputs': [],
                'z_values': [],
                'deltas': []
            })

    def relu(self, x):
        return max(0.0, x)

    def relu_derivative(self, x):
        return 1.0 if x > 0 else 0.0

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, inputs):
        """
        Perform a forward pass through the network.
        :param inputs: List of input values.
        :return: Output of the network.
        """
        activations = inputs
        for idx, layer in enumerate(self.layers):
            new_activations = []
            layer['inputs'] = activations  # Store inputs for backprop
            layer['z_values'] = []
            layer['outputs'] = []
            for j in range(len(layer['biases'])):
                activation = layer['biases'][j]
                for i in range(len(activations)):
                    activation += activations[i] * layer['weights'][i][j]
                layer['z_values'].append(activation)
                # Apply activation function
                if idx < len(self.layers) - 1:
                    # Hidden layers use ReLU
                    activated = self.relu(activation)
                else:
                    # Output layer uses linear activation for Q-values
                    activated = activation
                layer['outputs'].append(activated)
                new_activations.append(activated)
            activations = new_activations
        return activations

    def backward(self, target_outputs, learning_rate):
        """
        Perform backward propagation and update weights.
        :param target_outputs: Expected outputs.
        :param learning_rate: Learning rate for weight updates.
        """
        # Compute deltas for output layer
        last_layer = self.layers[-1]
        deltas = []
        for i in range(len(last_layer['outputs'])):
            output = last_layer['outputs'][i]
            z = last_layer['z_values'][i]
            target = target_outputs[i]
            # Compute derivative of loss w.r.t output
            error = output - target
            # Derivative of activation function (linear activation)
            derivative = 1.0
            delta = error * derivative
            deltas.append(delta)
        last_layer['deltas'] = deltas

        # Backpropagate deltas to hidden layers
        for idx in reversed(range(len(self.layers) - 1)):
            layer = self.layers[idx]
            next_layer = self.layers[idx + 1]
            deltas = []
            for i in range(len(layer['outputs'])):
                z = layer['z_values'][i]
                derivative = self.relu_derivative(z)
                error = 0.0
                for j in range(len(next_layer['deltas'])):
                    error += next_layer['deltas'][j] * next_layer['weights'][i][j]
                delta = error * derivative
                deltas.append(delta)
            layer['deltas'] = deltas

        # Update weights and biases
        for idx, layer in enumerate(self.layers):
            inputs = layer['inputs']
            for i in range(len(layer['weights'])):
                for j in range(len(layer['weights'][i])):
                    # Weight update
                    layer['weights'][i][j] -= learning_rate * layer['deltas'][j] * inputs[i]
            for j in range(len(layer['biases'])):
                # Bias update
                layer['biases'][j] -= learning_rate * layer['deltas'][j]

# Modified Agent Class with Learning Capability and Breeding Restrictions
class Agent:
    def __init__(self, agent_id, x, y, is_child=False):
        self.id = agent_id
        self.x = x
        self.y = y
        self.energy = INITIAL_ENERGY
        self.personality = self.generate_personality()
        self.status = 'alive'
        self.age = 0.0  # Changed to float to represent seconds
        self.gender = self.assign_gender()
        self.mutation = None
        self.color = self.assign_color()
        self.size = AGENT_SIZE if not is_child else CHILD_SIZE
        self.group = None  # New attribute for group affiliation
        self.nn = AdvancedNeuralNetwork(input_size=14, hidden_layers=[256, 256, 128], output_size=6)
        self.memory = []
        self.epsilon = 0.5  # Exploration rate
        # New attributes for reproduction
        self.children = []
        self.max_offspring = self.calculate_max_offspring()
        self.partners = set()  # Keep track of partners
        self.last_breed_time = -float('inf')  # Tracks the last time the agent bred

    def generate_personality(self):
        """
        Generate a personality based on the Big Five traits.
        Each trait is rated between 0 (low) to 1 (high).
        """
        return {
            'openness': random.uniform(0, 1),
            'conscientiousness': random.uniform(0, 1),
            'extraversion': random.uniform(0, 1),
            'agreeableness': random.uniform(0, 1),
            'neuroticism': random.uniform(0, 1)
        }

    def assign_color(self):
        openness = self.personality['openness']
        conscientiousness = self.personality['conscientiousness']
        extraversion = self.personality['extraversion']
        agreeableness = self.personality['agreeableness']
        neuroticism = self.personality['neuroticism']

        # Map personality traits to RGB color
        base_color = (
            int(AGENT_BASE_COLOR[0] * openness),
            int(AGENT_BASE_COLOR[1] * conscientiousness),
            int(AGENT_BASE_COLOR[2] * extraversion)
        )

        # Modify color based on agreeableness and neuroticism
        base_color = (
            min(255, base_color[0] + int(AGENT_BASE_COLOR[0] * agreeableness)),
            min(255, base_color[1] + int(AGENT_BASE_COLOR[1] * neuroticism)),
            min(255, base_color[2] + int(AGENT_BASE_COLOR[2] * agreeableness))
        )

        # If mutated, alter color based on mutation
        if self.mutation:
            mutation_color = MUTATION_COLORS.get(self.mutation, AGENT_BASE_COLOR)
            # Blend base color with mutation color
            blended_color = (
                (base_color[0] + mutation_color[0]) // 2,
                (base_color[1] + mutation_color[1]) // 2,
                (base_color[2] + mutation_color[2]) // 2
            )
            return blended_color
        return base_color

    def assign_gender(self):
        return random.choice(['male', 'female'])

    def calculate_max_offspring(self):
        """
        Calculate the maximum number of offspring based on resources and personality traits.
        """
        base_offspring = int(self.energy / 50)  # More energy allows for more offspring
        if self.gender == 'female':
            # Females may be limited by conscientiousness and agreeableness
            modifier = (self.personality['conscientiousness'] + self.personality['agreeableness']) / 2
        else:
            # Males may have more offspring if extraverted and open
            modifier = (self.personality['extraversion'] + self.personality['openness']) / 2
        max_offspring = max(1, int(base_offspring * modifier))
        return max_offspring

    def perceive_environment(self, world):
        # Find nearest food
        nearest_food = None
        min_food_dist = float('inf')
        for food in world.food:
            dist = math.hypot(self.x - food[0], self.y - food[1])
            if dist < min_food_dist:
                min_food_dist = dist
                nearest_food = food

        # Find nearest water
        nearest_water = None
        min_water_dist = float('inf')
        for water in world.water:
            dist = math.hypot(self.x - water[0], self.y - water[1])
            if dist < min_water_dist:
                min_water_dist = dist
                nearest_water = water

        # Find nearby agents
        nearby_agents = []
        for agent in world.agents:
            if agent.id != self.id and agent.status == 'alive':
                dist = math.hypot(self.x - agent.x, self.y - agent.y)
                if dist < 100:  # Consider agents within 100 units as nearby
                    nearby_agents.append(agent)

        # Normalize inputs
        norm_food_dist = min_food_dist / math.hypot(world.width, world.height)
        norm_water_dist = min_water_dist / math.hypot(world.width, world.height)
        norm_energy = self.energy / 100  # Normalize based on threshold
        norm_openness = self.personality['openness']
        norm_conscientiousness = self.personality['conscientiousness']
        norm_extraversion = self.personality['extraversion']
        norm_agreeableness = self.personality['agreeableness']
        norm_neuroticism = self.personality['neuroticism']

        inputs = [
            norm_food_dist,
            norm_water_dist,
            norm_energy,
            norm_openness,
            norm_conscientiousness,
            norm_extraversion,
            norm_agreeableness,
            norm_neuroticism,
            len(nearby_agents) / 10  # Number of nearby agents normalized
        ]

        return {
            'nearest_food': nearest_food,
            'min_food_dist': min_food_dist,
            'nearest_water': nearest_water,
            'min_water_dist': min_water_dist,
            'inputs': inputs,
            'nearby_agents': nearby_agents
        }

    def move(self, world, delta_time):
        # Perceive environment
        perception = self.perceive_environment(world)
        nearest_food = perception['nearest_food']
        min_food_dist = perception['min_food_dist']
        nearest_water = perception['nearest_water']
        min_water_dist = perception['min_water_dist']

        # Decide which resource to move towards
        if min_food_dist < min_water_dist:
            target_x, target_y = nearest_food
        else:
            target_x, target_y = nearest_water

        # Calculate direction towards the target
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)
        if distance > 0:
            direction_x = dx / distance
            direction_y = dy / distance
        else:
            direction_x, direction_y = 0, 0

        # Move towards the target
        speed = 100  # Units per second
        self.x += direction_x * speed * delta_time
        self.y += direction_y * speed * delta_time

        # Keep within world bounds
        self.x = max(0, min(world.width, self.x))
        self.y = max(0, min(world.height, self.y))

        # Attempt to consume resources directly
        interacted = self.consume_resources(world)

        return interacted

    def consume_resources(self, world):
        # Attempt to consume Food
        for food in world.food:
            distance = math.hypot(self.x - food[0], self.y - food[1])
            if distance < CONSUMPTION_RADIUS:
                self.energy += 50  # Increased energy gain
                world.food.remove(food)
                return True  # Successful interaction

        # Attempt to consume Water
        for water in world.water:
            distance = math.hypot(self.x - water[0], self.y - water[1])
            if distance < CONSUMPTION_RADIUS:
                self.energy += 30  # Increased energy gain
                world.water.remove(water)
                return True  # Successful interaction

        return False  # No interaction occurred

    def mutate(self):
        global mutation_counts
        if not self.mutation and random.random() < MUTATION_PROBABILITY:
            # Assign a mutation from A-Z
            available_mutations = [m for m in MUTATION_COLORS.keys() if mutation_counts[m] < 1000]  # Limit per mutation
            if available_mutations:
                self.mutation = random.choice(available_mutations)
                mutation_counts[self.mutation] += 1
                self.color = self.assign_color()

    def breed(self, partner, world, current_time):
        # Check if agents can breed based on max offspring and energy
        if self.gender == partner.gender:
            return  # Cannot breed with same gender

        if self.gender == 'female':
            mother, father = self, partner
        else:
            mother, father = partner, self

        # Females are limited by max_offspring
        if len(mother.children) >= mother.max_offspring:
            return

        # Males may compete; higher energy males have an advantage
        if father.energy < BREED_ENERGY_THRESHOLD:
            return

        # Proceed with breeding
        child_personality = {}
        for trait in self.personality:
            child_personality[trait] = (self.personality[trait] + partner.personality[trait]) / 2 + random.uniform(-0.05, 0.05)
            # Ensure traits remain within [0,1]
            child_personality[trait] = max(0, min(1, child_personality[trait]))
        child = Agent(world.next_agent_id, (self.x + partner.x) / 2, (self.y + partner.y) / 2, is_child=True)
        child.personality = child_personality
        child.color = child.assign_color()
        child.mutate()  # Potential mutation during birth
        world.agents.append(child)
        world.next_agent_id += 1

        # Deduct energy after breeding
        self.energy -= 40
        partner.energy -= 40

        # Add child to parents' list
        mother.children.append(child)
        father.children.append(child)

        # Add partners
        mother.partners.add(father)
        father.partners.add(mother)

        # Update last_breed_time for both parents
        self.last_breed_time = current_time
        partner.last_breed_time = current_time

        # Check for twins
        if random.random() < TWINS_PROBABILITY:
            # Create a second child (twins)
            twin = Agent(world.next_agent_id, (self.x + partner.x) / 2 + random.uniform(-5, 5), (self.y + partner.y) / 2 + random.uniform(-5, 5), is_child=True)
            twin.personality = child_personality.copy()
            twin.color = twin.assign_color()
            twin.mutate()  # Potential mutation during birth
            world.agents.append(twin)
            world.next_agent_id += 1

            # Deduct additional energy for twins
            self.energy -= 20
            partner.energy -= 20

            # Add twin to parents' list
            mother.children.append(twin)
            father.children.append(twin)

    def update(self, world, current_time, delta_time):
        if self.status == 'alive':
            interacted = self.move(world, delta_time)

            # Interaction with nearby agents
            self.interact(world, current_time, delta_time)

            # Energy decreases based on time_scale
            self.energy -= ENERGY_DECREMENT * delta_time

            if not interacted:
                # If didn't consume any resources, energy decreases more
                self.energy -= ENERGY_DECREMENT * delta_time  # Additional decrement

            if self.energy <= 0:
                self.status = 'dead'
                # Remove from group if dead
                if self.group:
                    self.group.remove_member(self)

            else:
                # Increment age based on time_scale
                self.age += delta_time
                if self.age <= ADULT_AGE:
                    self.size = CHILD_SIZE + (AGENT_SIZE - CHILD_SIZE) * (self.age / ADULT_AGE)
                else:
                    self.size = AGENT_SIZE

                # Recalculate max offspring based on energy
                self.max_offspring = self.calculate_max_offspring()

                # Attempt to breed
                for agent in world.agents:
                    if agent.id != self.id and agent.status == 'alive':
                        distance = math.hypot(self.x - agent.x, self.y - agent.y)
                        if distance < CONSUMPTION_RADIUS:
                            if self.can_breed_with(agent, current_time):
                                self.breed(agent, world, current_time)
                                break  # Only breed with one agent per update

                # Attempt mutation
                self.mutate()

    def can_breed_with(self, agent, current_time):
        if self.gender == agent.gender:
            return False
        if self.gender == 'female' and len(self.children) >= self.max_offspring:
            return False
        if agent.gender == 'female' and len(agent.children) >= agent.max_offspring:
            return False
        if self.energy < BREED_ENERGY_THRESHOLD or agent.energy < BREED_ENERGY_THRESHOLD:
            return False
        # Check cooldown (assuming cooldown is in seconds)
        if current_time - self.last_breed_time < 90 or current_time - agent.last_breed_time < 90:
            return False
        return True

    def interact(self, world, current_time, delta_time):
        # Find nearby agents
        nearby_agents = []
        for agent in world.agents:
            if agent.id != self.id and agent.status == 'alive':
                distance = math.hypot(self.x - agent.x, self.y - agent.y)
                if distance < CONSUMPTION_RADIUS * 2:  # Interaction radius
                    nearby_agents.append(agent)

        for other_agent in nearby_agents:
            # Prepare inputs for neural network
            inputs = self.prepare_interaction_inputs(other_agent, world)

            # Get decision from neural network
            outputs = self.nn.forward(inputs)

            # Interpret outputs
            action, action_index = self.decide_action(outputs)

            # Execute action
            reward = self.execute_action(action, other_agent, world)

            # Compute target outputs
            target_outputs = outputs.copy()
            target_outputs[action_index] = reward  # Set target for taken action

            # Perform backward pass
            learning_rate = 0.01
            self.nn.backward(target_outputs, learning_rate)

    def prepare_interaction_inputs(self, other_agent, world):
        # Normalize values between 0 and 1
        norm_energy = self.energy / 100
        norm_other_energy = other_agent.energy / 100
        norm_distance = math.hypot(self.x - other_agent.x, self.y - other_agent.y) / math.hypot(world.width, world.height)
        # Difference in personality traits
        openness_diff = abs(self.personality['openness'] - other_agent.personality['openness'])
        conscientiousness_diff = abs(self.personality['conscientiousness'] - other_agent.personality['conscientiousness'])
        extraversion_diff = abs(self.personality['extraversion'] - other_agent.personality['extraversion'])
        agreeableness_diff = abs(self.personality['agreeableness'] - other_agent.personality['agreeableness'])
        neuroticism_diff = abs(self.personality['neuroticism'] - other_agent.personality['neuroticism'])

        # Group affiliation (1 if same group, 0 otherwise)
        same_group = 1 if self.group and self.group == other_agent.group else 0

        # Inputs for the neural network
        inputs = [
            norm_energy,
            norm_other_energy,
            norm_distance,
            openness_diff,
            conscientiousness_diff,
            extraversion_diff,
            agreeableness_diff,
            neuroticism_diff,
            self.personality['openness'],
            self.personality['conscientiousness'],
            self.personality['extraversion'],
            self.personality['agreeableness'],
            self.personality['neuroticism'],
            same_group
        ]
        return inputs

    def decide_action(self, outputs):
        # Outputs are action values (Q-values)
        actions = ['kill', 'counterattack', 'share', 'help', 'stay_neutral', 'form_group']
        if random.random() < self.epsilon:
            # Explore: choose a random action
            action_index = random.randint(0, len(actions) - 1)
        else:
            # Exploit: choose action with highest Q-value
            action_index = outputs.index(max(outputs))
        action = actions[action_index]
        return action, action_index

    def execute_action(self, action, other_agent, world):
        reward = 0
        if action == 'kill':
            # Attempt to kill the other agent
            if other_agent.group != self.group or self.group is None:
                other_agent.status = 'dead'
                other_agent.energy = 0
                # Gain some energy from the kill
                self.energy += 20
                reward = 20  # Positive reward for killing
                # Remove from group if dead
                if other_agent.group:
                    other_agent.group.remove_member(other_agent)
            else:
                reward = -10  # Penalty for attacking group member
        elif action == 'counterattack':
            # Implement counterattack logic if desired
            reward = -5  # Small penalty
        elif action == 'share':
            # Share energy with the other agent
            energy_shared = min(10, self.energy * 0.1)
            self.energy -= energy_shared
            other_agent.energy += energy_shared
            reward = 5  # Positive reward for cooperation
        elif action == 'help':
            # Help the other agent by boosting their energy
            energy_boost = 5
            self.energy -= energy_boost
            other_agent.energy += energy_boost
            reward = 5  # Positive reward for helping
        elif action == 'stay_neutral':
            # Do nothing
            reward = 0
        elif action == 'form_group':
            # Attempt to form a group
            if self.group is None and other_agent.group is None:
                # Create new group
                new_group = Group(world.next_group_id)
                world.next_group_id += 1
                new_group.add_member(self)
                new_group.add_member(other_agent)
                world.groups.append(new_group)
                reward = 10  # Positive reward for forming a group
            elif self.group is not None and other_agent.group is None:
                # Add other agent to self's group
                self.group.add_member(other_agent)
                reward = 5  # Positive reward
            elif self.group is None and other_agent.group is not None:
                # Add self to other agent's group
                other_agent.group.add_member(self)
                reward = 5  # Positive reward
            elif self.group != other_agent.group:
                # Merge groups
                self.group.merge_with(other_agent.group, world)
                reward = 2  # Small positive reward
        return reward

    def draw(self, surface, zoom, offset_x, offset_y):
        # Draw a simple human-like figure
        adjusted_x = self.x * zoom + offset_x
        adjusted_y = self.y * zoom + offset_y
        size = self.size * zoom

        # Head
        pygame.draw.circle(surface, self.color, (int(adjusted_x), int(adjusted_y - size * 1.5)), int(max(size * 0.5, 1)))
        # Body
        pygame.draw.line(surface, self.color, (int(adjusted_x), int(adjusted_y - size)),
                         (int(adjusted_x), int(adjusted_y)),
                         int(max(int(size * 0.1), 1)))
        # Arms
        pygame.draw.line(surface, self.color, (int(adjusted_x - size), int(adjusted_y - size * 0.75)),
                         (int(adjusted_x + size), int(adjusted_y - size * 0.75)),
                         int(max(int(size * 0.1), 1)))
        # Legs
        pygame.draw.line(surface, self.color, (int(adjusted_x), int(adjusted_y)),
                         (int(adjusted_x - size * 0.5), int(adjusted_y + size)),
                         int(max(int(size * 0.1), 1)))
        pygame.draw.line(surface, self.color, (int(adjusted_x), int(adjusted_y)),
                         (int(adjusted_x + size * 0.5), int(adjusted_y + size)),
                         int(max(int(size * 0.1), 1)))

        # Draw gender symbol
        gender_font_size = int(max(size * 0.8, 8))  # Ensure minimum font size
        gender_font = pygame.font.SysFont('Arial', gender_font_size)
        gender_symbol = '♂' if self.gender == 'male' else '♀'
        gender_surface = gender_font.render(gender_symbol, True, BLACK)
        symbol_x = int(adjusted_x - size)
        symbol_y = int(adjusted_y - size * 1.5 - gender_font_size)
        surface.blit(gender_surface, (symbol_x, symbol_y))

        # Draw mutation label if any
        if self.mutation:
            mutation_surface = FONT.render(self.mutation, True, BLACK)
            surface.blit(mutation_surface, (int(adjusted_x) + size * 0.5, int(adjusted_y) - size * 1.5 - 15))

        # Optional: Draw personality traits as bars or indicators
        traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
        for idx, trait in enumerate(traits):
            trait_value = self.personality[trait]
            bar_length = int(trait_value * size)
            pygame.draw.rect(surface, BLACK, pygame.Rect(int(adjusted_x - size), int(adjusted_y) + idx * 5, bar_length, 3))

# Group Class
class Group:
    def __init__(self, group_id):
        self.id = group_id
        self.members = []

    def add_member(self, agent):
        self.members.append(agent)
        agent.group = self

    def remove_member(self, agent):
        if agent in self.members:
            self.members.remove(agent)
            agent.group = None

    def merge_with(self, other_group, world):
        # Merge other_group into this group
        for member in other_group.members:
            self.add_member(member)
        # Remove other_group from world's group list
        if other_group in world.groups:
            world.groups.remove(other_group)

# World Class with Groups and Time Tracking
class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []
        self.groups = []  # New attribute to hold groups
        self.food = []
        self.water = []
        self.next_agent_id = 1
        self.next_group_id = 1  # New attribute for group IDs
        self.current_time = 0.0  # Tracks simulation time in seconds
        self.spawn_initial_agents(15)
        self.spawn_resources()

    def spawn_initial_agents(self, count):
        for _ in range(count):
            x = random.uniform(100, self.width - 100)
            y = random.uniform(100, self.height - 100)
            agent = Agent(self.next_agent_id, x, y)
            self.agents.append(agent)
            self.next_agent_id += 1

    def spawn_resources(self):
        for _ in range(INITIAL_FOOD):
            self.spawn_food()
        for _ in range(INITIAL_WATER):
            self.spawn_water()

    def spawn_food(self):
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        self.food.append((x, y))

    def spawn_water(self):
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        self.water.append((x, y))

    def update(self, delta_time):
        self.current_time += delta_time  # Increment simulation time

        for agent in self.agents[:]:
            agent.update(self, self.current_time, delta_time)

        # Remove dead agents
        self.agents = [agent for agent in self.agents if agent.status != 'dead']

        # Remove empty groups
        self.groups = [group for group in self.groups if group.members]

        # Regenerate resources
        if random.random() < RESOURCE_REGENERATION_RATE:
            self.spawn_food()
        if random.random() < RESOURCE_REGENERATION_RATE:
            self.spawn_water()

    def draw(self, surface, zoom, offset_x, offset_y):
        # Draw Food
        for food in self.food:
            adjusted_x = food[0] * zoom + offset_x
            adjusted_y = food[1] * zoom + offset_y
            pygame.draw.rect(surface, FOOD_COLOR,
                             pygame.Rect(int(adjusted_x), int(adjusted_y), max(int(5 * zoom), 1), max(int(5 * zoom), 1)))

        # Draw Water
        for water in self.water:
            adjusted_x = water[0] * zoom + offset_x
            adjusted_y = water[1] * zoom + offset_y
            pygame.draw.rect(surface, WATER_COLOR,
                             pygame.Rect(int(adjusted_x), int(adjusted_y), max(int(5 * zoom), 1), max(int(5 * zoom), 1)))

        # Draw Agents
        for agent in self.agents:
            if agent.status == 'alive':
                agent.draw(surface, zoom, offset_x, offset_y)

        # Draw groups (Updated to tightly circle group members)
        for group in self.groups:
            group_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            member_positions = [(agent.x * zoom + offset_x, agent.y * zoom + offset_y) for agent in group.members]
            if member_positions:
                # Compute center
                center_x = sum(pos[0] for pos in member_positions) / len(member_positions)
                center_y = sum(pos[1] for pos in member_positions) / len(member_positions)
                # Compute radius as the maximum distance from center to any member position plus padding
                radius = max(math.hypot(pos[0] - center_x, pos[1] - center_y) for pos in member_positions) + AGENT_SIZE * zoom
                radius = int(radius)
                pygame.draw.circle(surface, group_color, (int(center_x), int(center_y)), radius, 2)

    def count_genders(self):
        males = len([agent for agent in self.agents if agent.status == 'alive' and agent.gender == 'male'])
        females = len([agent for agent in self.agents if agent.status == 'alive' and agent.gender == 'female'])
        return males, females

    def count_mutations(self):
        return mutation_counts.copy()

# Function to scale GUI elements
def scale_gui(scale_factor):
    # Prevent scaling beyond limits
    global gui_scale_factor
    gui_scale_factor = max(MIN_GUI_SCALE, min(MAX_GUI_SCALE, scale_factor))
    # Apply scaling to each GUI element based on original properties
    for key, rect in original_gui_elements.items():
        element = globals()[key]
        new_rect = pygame.Rect(
            rect.x * gui_scale_factor,
            rect.y * gui_scale_factor,
            rect.width * gui_scale_factor,
            rect.height * gui_scale_factor
        )
        element.set_relative_rect(new_rect)
    # Update the UI manager's window size if needed
    # Note: pygame_gui does not support dynamic scaling out of the box,
    # so elements are manually resized and repositioned.

# Function to update the speed label
def update_speed_label():
    speed_label.set_text(f'Speed: {time_scale}x')

# Initialize World
world = World(WORLD_WIDTH, WORLD_HEIGHT)

# Main Loop
running = True
while running:
    time_delta = clock.tick(FPS) / 1000.0  # Delta time in seconds

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        # Handle GUI events
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == start_button:
                running_simulation = True
                try:
                    simulation_duration = int(duration_input.get_text())
                except ValueError:
                    simulation_duration = None
            elif event.ui_element == stop_button:
                running_simulation = False
            elif event.ui_element == restart_button:
                # Reset mutation counts
                mutation_counts = {chr(i): 0 for i in range(65, 91)}
                # Reset world
                world = World(WORLD_WIDTH, WORLD_HEIGHT)
                running_simulation = False
            elif event.ui_element == speed_up_button:
                if time_scale < MAX_TIME_SCALE:
                    time_scale += 1
                    update_speed_label()
            elif event.ui_element == slow_down_button:
                if time_scale > MIN_TIME_SCALE:
                    time_scale -= 1
                    update_speed_label()

        # Handle Zoom and Pan
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                dragging = True
                last_mouse_pos = event.pos
            elif event.button == 4:  # Scroll up
                zoom_factor *= 1.1
            elif event.button == 5:  # Scroll down
                zoom_factor /= 1.1

        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:  # Left click
                dragging = False

        elif event.type == MOUSEMOTION:
            if dragging:
                dx, dy = event.rel
                offset_x += dx
                offset_y += dy

        elif event.type == KEYDOWN:
            if event.key == K_EQUALS or event.key == pygame.K_PLUS:
                # Increase GUI scale
                scale_gui(gui_scale_factor + SCALE_STEP)
            elif event.key == K_MINUS:
                # Decrease GUI scale
                scale_gui(gui_scale_factor - SCALE_STEP)

        manager.process_events(event)

    manager.update(time_delta)

    if running_simulation:
        # Update the world with scaled time_delta
        scaled_delta_time = time_delta * time_scale
        world.update(scaled_delta_time)
        if simulation_duration is not None:
            simulation_duration -= scaled_delta_time
            if simulation_duration <= 0:
                running_simulation = False

    # Drawing
    screen.fill(WHITE)
    world.draw(screen, zoom_factor, offset_x, offset_y)

    # Draw GUI
    manager.draw_ui(screen)

    # Optional: Display Simulation Stats
    agent_count = len([agent for agent in world.agents if agent.status == 'alive'])
    food_count = len(world.food)
    water_count = len(world.water)
    males, females = world.count_genders()
    mutations = world.count_mutations()

    # Prepare Mutation Stats
    mutation_stats = ' | '.join([f'{k}: {v}' for k, v in mutations.items() if v > 0])

    # Combine all stats
    stats_text = f'Agents: {agent_count} | Males: {males} | Females: {females} | Food: {food_count} | Water: {water_count}'
    stats_surface = FONT.render(stats_text, True, BLACK)
    screen.blit(stats_surface, (10, 60))

    # Display Mutation Stats
    mutation_surface = FONT.render(f'Mutations: {mutation_stats}', True, BLACK)
    screen.blit(mutation_surface, (10, 80))

    pygame.display.flip()

pygame.quit()
sys.exit()
