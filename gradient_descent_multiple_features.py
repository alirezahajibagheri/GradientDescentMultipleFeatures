"""
Author: Alireza Hajibagheri
Email: nima.hajibagheri@gmail.com

Gradient descent for solving the following hypothesis:

h_theta(x) = theta_0 * x_0 + theta_1 * x_1 + ... + theta_n * x_n

where x_0 = 1 and cost function is defined as:

J_theta(theta) = 1/2m Σ (h_theta(x^i) - y^i)^2

Goal is to minimize J_theta (sum squared errors) and find optimal values theta_0, theta_1, ..., theta_n (n = number of features)
"""
import numpy as np
import pandas as pd


class GradientDescent:
    def __init__(self, x_data, y_data_array):
        """ Run gradient descent algorithm on input data
            :param x_data: input x values in form of a pandas dataframe with n columns where n is the number of features
            :param y_data_array: input y values in form of a numpy array
        """
        thetas = self.gradient_descent(x_data = x_data, y_data_array = y_data_array)
        print('Found optimal values for thetas: {0}'.format(thetas))

    def gradient_descent(self, x_data, y_data_array):
        """
            :param x_data: input x values in form of a pandas dataframe with n columns where n is the number of features
            :param y_data_array: input y values in form of a numpy array
            :return thetas: final estimated values for thetas
        """
        # Add x_0 column to the input dataframe (x_0 is always 1)
        column_names = list(x_data)
        x_data['x_0'] = 1.0
        column_names = ['x_0'] + column_names
        x_data = x_data[column_names]
        x_data_array = x_data.to_numpy()

        # Initialize variables
        sample_size = x_data.shape[0]
        feature_size = x_data.shape[1]
        thetas = np.array(feature_size * [0.0])
        learning_rate = 0.1
        total_iterations = 1000
        iter_count = 0

        while iter_count < total_iterations:

            # Get h_theta values (predictions)
            h_theta = self.hypothesis_function(thetas, x_data_array)

            # Calculate h_theta - y (predictions - real data)
            h_theta_y_diff = h_theta - y_data_array

            # Calculate (h_theta - y) * h_j where j is feature index
            h_theta_y_diff_times_x = np.matmul(h_theta_y_diff, x_data_array)

            # Get updated theta_j value where j is feature index
            # theta_j := theta_j - learning_rate * 1/sample_size * Σ(h_theta(x^i) - y^i) * x_j
            thetas = thetas - learning_rate * (1 / sample_size) * h_theta_y_diff_times_x

            # Update iter_count
            iter_count += 1

        # Rounding theta values
        thetas = [round(this_theta, 2) for this_theta in thetas]

        return thetas

    @staticmethod
    def hypothesis_function(thetas, x_data_array):
        """ Returns h_theta(x) using the following equation:
            h_theta(x) = Σ theta_i * x_i
            :param thetas: all theta values in form of a numpy array
            :param x_data_array: all features values as a numpy array
            :return: h_theta: h_theta(x) in form of a numpy array
        """
        # Calculate h_theta which is basically dot product of x values times thetas
        h_theta = np.matmul(x_data_array, thetas)

        return h_theta


# Testing the code (code should print out teta_0 = 0 and teta_1 = 1 for this sample dataset)
# Sample data (x)
data_tuples = [(0.0, 0.0),
               (1.0, 1.0),
               (2.0, 2.0),
               (3.0, 3.0)]
x_data = pd.DataFrame([this_tuple for this_tuple in data_tuples], columns = ['x_1', 'x_2'])

# Sample data (y)
y_data_array = np.array([0.0, 1.0, 2.0, 3.0])

# Get the best estimated theta values using gradient descent
new_gradient_descent = GradientDescent(x_data = x_data, y_data_array = y_data_array)
