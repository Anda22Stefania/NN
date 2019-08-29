import torch
import numpy as np
import matplotlib.pyplot as plt

x_train_init = np.array([[3.3], [4.4], [5.5], [6.71], [4.168], [9.779], [6.182]], dtype="float32")
y_train_init = np.array([[1.7], [2.76], [2.09], [3.19], [1.573], [3.366], [2.596]], dtype="float32")
plt.plot(x_train_init, y_train_init, "ro", label="Original data")
plt.show()
# Convert the arrays to tensors
x_train = torch.from_numpy(x_train_init)
y_train = torch.from_numpy(y_train_init)
# The requires_grad attribute is false (makes sense because the data set doesn't
# have to be trained)

# Create the Neural Network
input_size = 1  # The input is represented by 1 feature
hidden_size = 100  # Have 100 units in a layer
output_size = 1  # The output is represented by 1 result
learning_rate = 1e-6

# Initialize the weights matrices for each layer: eg: input_size x hidden_size
w1 = torch.rand(input_size, hidden_size, requires_grad=True)
w2 = torch.rand(hidden_size, output_size, requires_grad=True)

epochs = 301

# Train the model
for epoch in range(1, epochs):
    # First multiply the input by the first set of weights (mm is matrix multiplication)
    # Then clamp the negative numbers to 0: accepted min is 0 (ReLU activation function)
    # Finally, multiply the output by the second set of weights => y predicted
    y_pred = x_train.mm(w1).clamp(min=0).mm(w2)
    # Compute the mean squared error
    loss = (y_pred - y_train).pow(2).sum()

    # Print the loss every 50 iterations to see how it converges
    if epoch % 50 == 0:
        print(epoch, loss.item())  # loss.item() gets the scalar value held in loss

    # After each forward pass do a backward pass
    # Loss is computed with respect to all tensors which have requires_grad=True
    loss.backward()  # Uses the autograd library

    # After the gradients have been computed, update the weights
    with torch.no_grad():  # Stop autograd tracking history
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # Reset gradients to 0
        w1.grad.zero_()
        w2.grad.zero_()

# Use the model for prediction
x_train_tensor = torch.from_numpy(x_train_init)
y_predicted_tensor = x_train_tensor.mm(w1).clamp(min=0).mm(w2)

#Remove the tensor from the computation graph and turn into numpy array
predicted = y_predicted_tensor.detach().numpy()

plt.plot(x_train_init, y_train_init, "ro", label="Original data")
plt.plot(x_train_init, predicted, label="Fitted line")
plt.legend()
plt.show()

