import matplotlib.pyplot as plt

# Define your function f(x)
def f(x):
    return x**2  # Replace this with your actual function

# Number of iterations
num_iterations = 10

# Generate x values and corresponding f(x) values
x_values = range(1, num_iterations + 1)
f_values = [f(x) for x in x_values]

print(x_values)
print(f_values)
# Plotting the graph
plt.plot(x_values, f_values, marker='o', linestyle='-', color='blue', label='f(x) vs Iteration')

# Adding labels and title
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('Graph: f(x) vs Iteration')

# Adding legend
plt.legend()

# Save the graph as an image file (e.g., PNG)
plt.savefig('./graph/function_iteration_plot.png')
