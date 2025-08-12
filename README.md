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

### Name: SHARIKA.R

### Register Number: 212223230204

```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(71)
x=torch.linspace(1,50,50).reshape(-1,1)
e=torch.randint(-8,9,(50,1),dtype=torch.float)
y=2*x+1+e

plt.scatter(x,y,color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Generated Data for Linear Regression")
plt.show()

class Model(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.linear=nn.Linear(in_features,out_features)
  def forward(self,x):
    return self.linear(x)

torch.manual_seed(59)
model=Model(1,1)

initial_weight=model.linear.weight.item()
initial_bias=model.linear.bias.item()
print("\nName: SHARIKA. R")
print("Register No: 212223230204")
print(f"Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n")

loss_function=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.001)

epochs=100
losses=[]

for epoch in range(1,epochs+1):
  optimizer.zero_grad()
  y_pred=model(x)
  loss=loss_function(y_pred,y)
  losses.append(loss.item())
  loss.backward()
  optimizer.step()
  print(f"epoch: {epoch:2} loss: {loss.item():10.8f}"
        f"weight: {model.linear.weight.item():10.8f}"
        f"bias: {model.linear.bias.item():10.8f}")
plt.plot(range(epochs),losses,color='blue')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss curve')
plt.show()

x1=torch.tensor([x.min().item(),x.max().item()])
y1=x1 * model.linear.weight.item()+model.linear.bias.item()

plt.scatter(x,y,label='Original Data')
plt.plot(x1,y1,'r',label='Best-Fit Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best_Fit Line')
plt.legend()
plt.show()

x_new=torch.tensor([[120.0]])
y_pred_new=model(x_new).item()
print('\nName: SHARIKA. R')
print('Register No: 212223230204')
print(f'\nPrediction for x=120: {y_pred_new:.8f}')
```

### Dataset Information
<img width="836" height="629" alt="image" src="https://github.com/user-attachments/assets/b11e4418-55fe-4ad3-ba2d-a24d960bf8cc" />
<img width="585" height="87" alt="image" src="https://github.com/user-attachments/assets/11108876-38f5-4b8c-a11a-02e41a744ce1" />

### OUTPUT
Training Loss Vs Iteration Plot
<img width="823" height="624" alt="image" src="https://github.com/user-attachments/assets/be5bf568-09e9-4d56-acfe-85f6bdcf75e9" />
Best Fit line plot
<img width="799" height="622" alt="image" src="https://github.com/user-attachments/assets/770c1aee-c882-40f1-9da3-49b5d56f3021" />

### New Sample Data Prediction
<img width="398" height="120" alt="image" src="https://github.com/user-attachments/assets/3ea9fab7-b636-4458-a3cc-ff778a71bfe7" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
