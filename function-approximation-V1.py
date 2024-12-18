import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


#define a function 
f = lambda n: n**2+3*n+2

#defince the range of the data 
x = [i for i in range(-150,150,3)]
y = [f(n) for n in x]

# Example data points
X = np.array(x, dtype=np.float32)
Y = np.array(y, dtype=np.float32)


#normalize X,Y to have Mean = 0 and STD = 1 
X_mean, X_std = X.mean(), X.std()
Y_mean, Y_std = Y.mean(), Y.std()

X_normalized = (X - X_mean) / X_std
Y_normalized = (Y - Y_mean) / Y_std

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_normalized).unsqueeze(1)  # Shape: (N, 1)
Y_tensor = torch.tensor(Y_normalized).unsqueeze(1)  # Shape: (N, 1)

#hyparameters 
epochs = 18_000
n = 64
learning_rate =5e-4 #Don't increase too much the learning rate . The model will not be able to learn. 

# Define a simple feedforward neural network
# You can add more layers based of the complex of the funtion 
# 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()     
        self.fc = nn.Sequential(
            nn.Linear(1, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Instantiate the model, loss function, and optimizer
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop

for epoch in range(epochs):
    # Forward pass
    predictions = model(X_tensor)
    loss = criterion(predictions, Y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.8f}")

    if epoch == 5_000:
        optimizer.param_groups[0]['lr'] = 1e-5 #Personla Opinion always decrease the learning rate for better result 
        print("change learning rate")

# Evaluate the model
x_new = np.linspace(min(X), max(X), 100, dtype=np.float32)
x_new_tensor = torch.tensor(x_new).unsqueeze(1)
y_new = model(x_new_tensor).detach().numpy()

with torch.no_grad():
    num = 20.0 #test the model with a number it hasn't see before
    num = (num-X_mean) /X_std #normalize the number 
    test = torch.tensor([num], dtype=torch.float32)
    pred = model(test).detach().numpy()
    pred = (pred*Y_std) +Y_mean #unormalize the y so you get the true value 
    print(f"{pred=} | {f(20)=}")


