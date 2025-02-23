import numpy as np
import matplotlib.pyplot as plt

# Function to compute the gradients of the cost function with respect to m and b
def compute_gradients(m, b, X, Y):
    """
    Computes the gradients (derivatives) of the Mean Squared Error (MSE) cost function with respect to m and b.

    Parameters:
    - m: Slope of the linear regression model
    - b: Intercept of the linear regression model
    - X: Input feature values
    - Y: Actual target values

    Returns:
    - d_m: Gradient of the cost function with respect to m
    - d_b: Gradient of the cost function with respect to b
    """
    n = len(X)  # Number of data points
    Y_pred = m * X + b  # Predicted values based on current m and b
    d_m = (-2/n) * np.sum(X * (Y - Y_pred))  # Gradient with respect to m
    d_b = (-2/n) * np.sum(Y - Y_pred)  # Gradient with respect to b
    return d_m, d_b

# Gradient Descent function to optimize m and b
def gradient_descent(X, Y, m_init=-1, b_init=1, learning_rate=0.1, max_iter=1000, tolerance=1e-3):
    """
    Performs gradient descent to find the optimal values of m and b for linear regression.

    Parameters:
    - X: Input feature values
    - Y: Actual target values
    - m_init: Initial value of m (slope)
    - b_init: Initial value of b (intercept)
    - learning_rate: Step size for gradient descent updates
    - max_iter: Maximum number of iterations for gradient descent
    - tolerance: Threshold to stop the algorithm when the cost change is below this value

    Returns:
    - m: Final optimized value of m
    - b: Final optimized value of b
    - cost_history: History of cost values at each iteration
    - m_values: List of m values during the optimization
    - b_values: List of b values during the optimization
    """
    m, b = m_init, b_init  # Initialize m and b
    cost_history = []  # To store the cost at each iteration
    m_values, b_values = [m], [b]  # To store the intermediate m and b values
    
    # Perform gradient descent for the specified number of iterations
    for _ in range(max_iter):
        d_m, d_b = compute_gradients(m, b, X, Y)  # Compute gradients
        
        m_new = m - learning_rate * d_m  # Update m using the gradient and learning rate
        b_new = b - learning_rate * d_b  # Update b using the gradient and learning rate
        
        # Calculate the new cost (Mean Squared Error)
        cost = np.mean((Y - (m_new * X + b_new))**2)
        cost_history.append(cost)  # Record the cost value for this iteration
        
        # Check if the cost change is small enough to stop the algorithm
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            break  # Stop if the change in cost is below the tolerance
        
        m, b = m_new, b_new  # Update m and b for the next iteration
        m_values.append(m)  # Store the updated m value
        b_values.append(b)  # Store the updated b value
    
    return m, b, cost_history, m_values, b_values

# Sample Data: X is the input feature and Y is the target value
X = np.array([1, 3])  # Input feature values
Y = np.array([3, 6])  # Actual target values

# Run Gradient Descent to optimize m and b for the given data
m_final, b_final, cost_history, m_values, b_values = gradient_descent(X, Y)

# Plotting the changes in m and b during the gradient descent optimization
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.plot(m_values, label='m values')  # Plot the values of m during the iterations
plt.plot(b_values, label='b values')  # Plot the values of b during the iterations
plt.xlabel('Iterations')  # Label for the x-axis
plt.ylabel('Parameter Values')  # Label for the y-axis
plt.legend()  # Display a legend to distinguish m and b values
plt.title('Convergence of m and b')  # Set the plot title
plt.show()  # Display the plot
