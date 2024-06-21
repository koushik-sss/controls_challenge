import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from . import BaseController

class Controller(BaseController):
    def __init__(self):
        super(Controller, self).__init__()

        self.error_history = []
        self.delta_error_history = []
        self.integral_error = 0

        self.moving_average_window = 70

        self.alpha = 0.101459
        self.ema_error = 0

        # Define the universe of discourse for each variable
        error = ctrl.Antecedent(np.arange(-50, 51, 0.01), 'error')
        delta_error = ctrl.Antecedent(np.arange(-50, 51, 0.01), 'delta_error')
        output = ctrl.Consequent(np.arange(-3, 3, 0.001), 'output')

        # Adjusting the membership functions for error
        error['negative_large'] = fuzz.trapmf(error.universe, [-50, -50, -12, -6])
        error['negative_medium'] = fuzz.trimf(error.universe, [-12, -6, 0])
        error['negative_small'] = fuzz.trimf(error.universe, [-6, -3, 0])
        error['zero'] = fuzz.trimf(error.universe, [-1.5, 0, 1.5])
        error['positive_small'] = fuzz.trimf(error.universe, [0, 3, 6])
        error['positive_medium'] = fuzz.trimf(error.universe, [0, 6, 12])
        error['positive_large'] = fuzz.trapmf(error.universe, [6, 12, 50, 50])

        # Adjusting the membership functions for delta_error
        delta_error['negative_large'] = fuzz.trapmf(delta_error.universe, [-50, -50, -12, -6])
        delta_error['negative_medium'] = fuzz.trimf(delta_error.universe, [-12, -6, 0])
        delta_error['negative_small'] = fuzz.trimf(delta_error.universe, [-6, -3, 0])
        delta_error['zero'] = fuzz.trimf(delta_error.universe, [-1.5, 0, 1.5])
        delta_error['positive_small'] = fuzz.trimf(delta_error.universe, [0, 3, 6])
        delta_error['positive_medium'] = fuzz.trimf(delta_error.universe, [6, 12, 18])
        delta_error['positive_large'] = fuzz.trapmf(delta_error.universe, [12, 18, 50, 50])

        # Adjusting the membership functions for output
        output['negative_high'] = fuzz.trimf(output.universe, [-3, -2.5, -2])
        output['negative_medium'] = fuzz.trimf(output.universe, [-2.5, -1.5, -1])
        output['negative_low'] = fuzz.trimf(output.universe, [-1.5, -1, -0.5])
        output['zero'] = fuzz.trimf(output.universe, [-0.5, 0, 0.5])
        output['positive_low'] = fuzz.trimf(output.universe, [0.5, 1, 1.5])
        output['positive_medium'] = fuzz.trimf(output.universe, [1, 1.5, 2.5])
        output['positive_high'] = fuzz.trimf(output.universe, [2, 2.5, 3])

        # Adjusting rules for better control
        rules = [
            ctrl.Rule(error['negative_large'] & delta_error['negative_large'], output['negative_high']),
            ctrl.Rule(error['negative_large'] & delta_error['negative_medium'], output['negative_high']),
            ctrl.Rule(error['negative_large'] & delta_error['negative_small'], output['negative_medium']),
            ctrl.Rule(error['negative_large'] & delta_error['zero'], output['negative_medium']),
            ctrl.Rule(error['negative_large'] & delta_error['positive_small'], output['negative_low']),
            ctrl.Rule(error['negative_large'] & delta_error['positive_medium'], output['negative_low']),
            ctrl.Rule(error['negative_large'] & delta_error['positive_large'], output['zero']),

            ctrl.Rule(error['negative_medium'] & delta_error['negative_large'], output['negative_high']),
            ctrl.Rule(error['negative_medium'] & delta_error['negative_medium'], output['negative_medium']),
            ctrl.Rule(error['negative_medium'] & delta_error['negative_small'], output['negative_medium']),
            ctrl.Rule(error['negative_medium'] & delta_error['zero'], output['negative_low']),
            ctrl.Rule(error['negative_medium'] & delta_error['positive_small'], output['negative_low']),
            ctrl.Rule(error['negative_medium'] & delta_error['positive_medium'], output['zero']),
            ctrl.Rule(error['negative_medium'] & delta_error['positive_large'], output['positive_low']),

            ctrl.Rule(error['negative_small'] & delta_error['negative_large'], output['negative_medium']),
            ctrl.Rule(error['negative_small'] & delta_error['negative_medium'], output['negative_medium']),
            ctrl.Rule(error['negative_small'] & delta_error['negative_small'], output['negative_low']),
            ctrl.Rule(error['negative_small'] & delta_error['zero'], output['negative_low']),
            ctrl.Rule(error['negative_small'] & delta_error['positive_small'], output['zero']),
            ctrl.Rule(error['negative_small'] & delta_error['positive_medium'], output['positive_low']),
            ctrl.Rule(error['negative_small'] & delta_error['positive_large'], output['positive_low']),

            ctrl.Rule(error['zero'] & delta_error['negative_large'], output['negative_low']),
            ctrl.Rule(error['zero'] & delta_error['negative_medium'], output['negative_low']),
            ctrl.Rule(error['zero'] & delta_error['negative_small'], output['zero']),
            ctrl.Rule(error['zero'] & delta_error['zero'], output['zero']),
            ctrl.Rule(error['zero'] & delta_error['positive_small'], output['zero']),
            ctrl.Rule(error['zero'] & delta_error['positive_medium'], output['positive_low']),
            ctrl.Rule(error['zero'] & delta_error['positive_large'], output['positive_low']),

            ctrl.Rule(error['positive_small'] & delta_error['negative_large'], output['negative_low']),
            ctrl.Rule(error['positive_small'] & delta_error['negative_medium'], output['zero']),
            ctrl.Rule(error['positive_small'] & delta_error['negative_small'], output['zero']),
            ctrl.Rule(error['positive_small'] & delta_error['zero'], output['positive_low']),
            ctrl.Rule(error['positive_small'] & delta_error['positive_small'], output['positive_low']),
            ctrl.Rule(error['positive_small'] & delta_error['positive_medium'], output['positive_medium']),
            ctrl.Rule(error['positive_small'] & delta_error['positive_large'], output['positive_medium']),

            ctrl.Rule(error['positive_medium'] & delta_error['negative_large'], output['zero']),
            ctrl.Rule(error['positive_medium'] & delta_error['negative_medium'], output['zero']),
            ctrl.Rule(error['positive_medium'] & delta_error['negative_small'], output['positive_low']),
            ctrl.Rule(error['positive_medium'] & delta_error['zero'], output['positive_low']),
            ctrl.Rule(error['positive_medium'] & delta_error['positive_small'], output['positive_medium']),
            ctrl.Rule(error['positive_medium'] & delta_error['positive_medium'], output['positive_medium']),
            ctrl.Rule(error['positive_medium'] & delta_error['positive_large'], output['positive_high']),

            ctrl.Rule(error['positive_large'] & delta_error['negative_large'], output['zero']),
            ctrl.Rule(error['positive_large'] & delta_error['negative_medium'], output['positive_low']),
            ctrl.Rule(error['positive_large'] & delta_error['negative_small'], output['positive_low']),
            ctrl.Rule(error['positive_large'] & delta_error['zero'], output['positive_medium']),
            ctrl.Rule(error['positive_large'] & delta_error['positive_small'], output['positive_medium']),
            ctrl.Rule(error['positive_large'] & delta_error['positive_medium'], output['positive_high']),
            ctrl.Rule(error['positive_large'] & delta_error['positive_large'], output['positive_high'])
        ]

        # Adding rules to handle zero delta_error more gracefully
        rules.extend([
            ctrl.Rule(error['negative_large'] & delta_error['zero'], output['negative_high']),
            ctrl.Rule(error['negative_medium'] & delta_error['zero'], output['negative_medium']),
            ctrl.Rule(error['negative_small'] & delta_error['zero'], output['negative_medium']),
            ctrl.Rule(error['zero'] & delta_error['negative_large'], output['negative_low']),
            ctrl.Rule(error['zero'] & delta_error['negative_medium'], output['negative_low']),
            ctrl.Rule(error['zero'] & delta_error['negative_small'], output['zero']),
            ctrl.Rule(error['zero'] & delta_error['positive_small'], output['zero']),
            ctrl.Rule(error['zero'] & delta_error['positive_medium'], output['positive_low']),
            ctrl.Rule(error['zero'] & delta_error['positive_large'], output['positive_low']),
            ctrl.Rule(error['positive_small'] & delta_error['zero'], output['positive_medium']),
            ctrl.Rule(error['positive_medium'] & delta_error['zero'], output['positive_medium']),
            ctrl.Rule(error['positive_large'] & delta_error['zero'], output['positive_high']),
        ])

        # Create the control system and simulation
        self.control_system = ctrl.ControlSystem(rules)
        self.control_system_simulation = ctrl.ControlSystemSimulation(self.control_system)


    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Compute the error between target and current lateral acceleration
        error = target_lataccel - current_lataccel
        self.error_history.append(error)

        # Maintain the error history within the moving average window size
        if len(self.error_history) > self.moving_average_window:
            self.error_history.pop(0)

        # Calculate the Exponential Moving Average (EMA) error
        if len(self.error_history) == 1:
            self.ema_error = self.error_history[0]
        else:
            self.ema_error = self.alpha * error + (1 - self.alpha) * self.ema_error

        # Calculate delta error if there is enough history, otherwise set to zero
        if len(self.error_history) > 1:
            delta_error = error - self.error_history[-2]
            delta_error = delta_error * delta_error * delta_error
            self.delta_error_history.append(delta_error)
        else:
            delta_error = 0

        # Maintain the delta error history within the moving average window size
        if len(self.delta_error_history) > self.moving_average_window:
            self.delta_error_history.pop(0)

        # Calculate the integral of the error
        self.integral_error = sum(self.error_history)

        # Strengthen the error term by combining proportional, integral, and derivative components
        strengthened_error = self.ema_error + 0.018 * self.integral_error

        # Input the strengthened error and the delta error into the fuzzy control system
        self.control_system_simulation.input['error'] = strengthened_error
        self.control_system_simulation.input['delta_error'] = delta_error

        # Compute the control system's output
        self.control_system_simulation.compute()

        # Return the output from the fuzzy control system simulation
        return self.control_system_simulation.output['output']
