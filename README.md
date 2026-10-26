# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Annapureddy kavya

### Register Number:212225240011

```python
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        #Include your code here
import torch
import torch.nn as nn  # Neural network module
import matplotlib.pyplot as plt  # For plotting
import numpy as np
%matplotlib inline
torch.manual_seed(71)  
X = torch.linspace(1, 50, 50).reshape(-1, 1)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
y = 2 * X + 1 + e
plt.scatter(X, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        #Include your code here
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        #Include your code here
        return self.linear(x)
torch.manual_seed(59)
model = Model(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("\nName:Annapureddy kavya ")
print("Register No:212225240011")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n')
epochs = 50
losses = []

for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, "
          f"Weight={model.linear.weight.item():.4f}, "
          f"Bias={model.linear.bias.item():.4f}")
plt.plot(range(1, epochs+1), losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss vs Epoch')
plt.show()
final_weight = model.linear.weight.item()
final_bias = model.linear.bias.item()
print("\nName:Annapureddy kavya ")
print("Register No: ")
print(f'\nFinal Weight: {final_weight:.8f}, Final Bias: {final_bias:.8f}')
x1 = torch.tensor([X.min().item(), X.max().item()])
y1 = x1 * final_weight + final_bias
plt.scatter(X, y, label="Original Data")
plt.plot(x1, y1, 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()
x_new = torch.tensor([[120.0]])
y_new_pred = model(x_new).item() 
print("\nName:Annapureddy kavya ")
print("Register No: 212225240011")
print(f"\nPrediction for x = 120: {y_new_pred:.8f}")
# Initialize the Model, Loss Function, and Optimizer

```

### Dataset Information
Include screenshot of the generated data
<img width="848" height="272" alt="image" src="https://github.com/user-attachments/assets/5ca4e2a2-1876-458e-8bd0-9b51c1cd4c57" />

### OUTPUT
Training Loss Vs Iteration Plot
<img width="722" height="570" alt="image" src="https://github.com/user-attachments/assets/188d64ae-5bb1-487d-ba9d-a5c3974bf732" />

Best Fit line plot
<img width="712" height="566" alt="image" src="https://github.com/user-attachments/assets/8566e7bc-6251-4481-adc2-28a07b0e1b77" />

Include your plot here
<img width="711" height="568" alt="image" src="https://github.com/user-attachments/assets/8e922197-42cc-42bb-bfbe-e5a61db5b226" />


### New Sample Data Prediction
Include your sample input and output here

<img width="772" height="288" alt="image" src="https://github.com/user-attachments/assets/918c01e1-da1f-4ce7-bef8-4b174ffca47b" />



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
