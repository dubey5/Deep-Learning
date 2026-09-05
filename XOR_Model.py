import torch
import torch.nn as nn

# Input data for XOR operation
X= torch.tensor([
    [0.0,0.0],
    [0.0,1.0],
    [1.0,0.0],
    [1.0,1.0]
])

# Correct answer for XOR rule
Y= torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

# Multi layer perceptron model for XOR operation
model= nn.Sequential(
    nn.Linear(2,4),  # First layer: 2 inputs → 4 hidden neurons
    nn.ReLU(),    # non-linear activation function
    nn.Linear(4,1),  # Second layer: 4 hidden-neuron outputs → 1 final output
    nn.Sigmoid()     # Activation function - converts final result to a value from 0 to 1
)

# Measure how wrong the prediction is
loss_function = nn.BCELoss()

# Updates the model weight while training
optimizer = torch.optim.Adam(model.parameters(),lr=0.05)

# Train the model
for _ in range(10000):
    prediction = model(X)
    loss = loss_function(prediction,Y)
    
    optimizer.zero_grad()       #clear old gradients
    loss.backward()             #calculate how weights caused the error
    optimizer.step()            # slightly improve weights


Z= torch.tensor([
    [1.0,1.0]
])

# Test the model
with torch.no_grad():
    predictions = model(Z)

print(predictions)
print((predictions > 0.5).int())




