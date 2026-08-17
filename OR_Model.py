import torch
import torch.nn as nn

# Training Examples [Two Inputs]
X= torch.tensor(
    [
        [0.0,0.0],
        [0.0,1.0],
        [1.0,0.0],
        [1.0,1.0]
    ]
)

# Correct answer for OR rule
Y= torch.tensor(
    [
        [0.0],
        [1.0],
        [1.0],
        [1.0],
    ]
)

# One neuron: 2 inputs -> 1 output
model= nn.Sequential(
    nn.Linear(2,1),
    nn.Sigmoid()
)

# Measure how wrong the prediction is
loss_function = nn.BCELoss()

# Updates the model weight while training
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

# Train the model
for _ in range(1000):
    prediction = model(X)
    loss = loss_function(prediction,Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Setup successful! Initial Loss: {loss.item():.4f}")

# Test the model

with torch.no_grad():
    predictions = model(X)

print(predictions)
print((predictions > 0.5).int())