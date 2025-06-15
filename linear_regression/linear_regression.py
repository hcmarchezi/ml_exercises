import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


# Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(
            in_features=input_dim,
            out_features=output_dim)

    def forward(self, x):
        return self.linear(x)


def calculate_mean_std(data: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
    return data.mean(dim=0, keepdim=True), data.std(dim=0, keepdim=True)


def normalize_data(data: torch.tensor, data_mean: torch.float32, data_std: torch.float32, epsilon: torch.float32 = 1e-8) -> torch.tensor:
    return (data - data_mean) / (data_std + epsilon)


# Read housing price data as csv
csv_file_path = os.path.dirname(os.path.realpath(__file__)) + '/housing_prices_dataset.csv'
df = pd.read_csv(csv_file_path).dropna()
x_data_df = df[['area', 'bathrooms', 'bedrooms', 'stories']]
y_data_df = df['price']
x_data = torch.tensor(x_data_df.values, dtype=torch.float32)
y_data = torch.tensor(y_data_df.values, dtype=torch.float32).unsqueeze(1)
x_mean, x_std = calculate_mean_std(x_data)
x_data = normalize_data(x_data, data_mean=x_mean, data_std=x_std)


# Instantiate Model, Loss Function, and Optimizer
model = LinearRegression(input_dim=4, output_dim=1)
loss_criteria = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Training the Model
num_epochs = 500
loss_history = []
loss_prev = torch.finfo(torch.float32).max
loss_stop_criteria = 150000

for epoch in range(num_epochs):
    model.train()
    y_predicted = model(x_data)

    loss = loss_criteria(y_predicted, y_data)
    loss_diff = loss_prev - loss
    loss_prev = loss
    loss_history.append(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"epoch={epoch} loss={loss} loss_diff={loss_diff}")
    if loss_diff < loss_stop_criteria or loss > loss_prev:
        print("stop training !!")
        break


# Plot train loss history
loss_history = [float(loss_item) for loss_item in loss_history]
plt.plot(loss_history)
plt.ylabel('loss')
plt.show()



# Make a prediction for new data
model.eval()
#         'area', 'bathrooms', 'bedrooms', 'stories'
new_x = [[  6000,          1,          2,         1],
         [  7000,          2,          2,         2],
         [  6500,          1,          2,         2],
         [  5000,          1,          1,         1]]
new_x = normalize_data(torch.tensor(new_x), data_mean=x_mean, data_std=x_std)
with torch.no_grad():
    predictions_for_new_x = model(new_x)

print("Prediction results for new data points:")
for i, x_val in enumerate(new_x):
    x_display = x_val.numpy()
    print(f"X: [{x_display}] -> Predicted Y: {predictions_for_new_x[i].item():.2f}")



