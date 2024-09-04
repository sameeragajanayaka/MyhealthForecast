import os
import io
import matplotlib
matplotlib.use('Agg')
from flask import Flask,request,send_file
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
app=Flask(__name__)



@app.route('/')
def hello():
    return "Hello, Flask!"

@app.route('/api')
def get():
    # str(request.args['Male'])
    # url = 'https://i0.wp.com/theperfectcurry.com/wp-content/uploads/2021/09/PXL_20210830_005055409.PORTRAIT.jpg'
    url=request.args['url']
    csv_id = request.args['id']
    # Fetch the image from the URL

    response = requests.get(url)
    save_folder = 'csv'  # Adjust the save folder path as needed
    save_name = 'calorie_'+csv_id+'.csv'
    if response.status_code == 200:
        # Create the save folder if it does not exist
        os.makedirs(save_folder, exist_ok=True)

        # Determine the image file name
        if save_name is None:
            save_name = os.path.basename(url)

        # Construct the full save path
        save_path = os.path.join(save_folder, save_name)

        # Write the image content to the file
        with open(save_path, 'wb') as file:
            file.write(response.content)

    # Load the dataset
    file_path = 'csv/calorie_'+csv_id+'.csv'  # Replace with your actual file path
    data = pd.read_csv(file_path)

    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Ensure the data is sorted by date
    data = data.sort_values('date')

    # Extract the calorie values
    calories = data['calorie'].values
    dates = data['date'].values

    # Normalize the data
    calories_mean = np.mean(calories)
    calories_std = np.std(calories)
    calories_normalized = (calories - calories_mean) / calories_std

    # Prepare the data
    def create_dataset(data, time_step=1, prediction_horizon=2):
        X, Y = [], []
        for i in range(len(data) - time_step - prediction_horizon + 1):
            a = data[i:(i + time_step)]
            X.append(a)
            Y.append(data[i + time_step + prediction_horizon - 1])
        return np.array(X), np.array(Y)

    time_step = 6  # Number of previous time steps to consider
    prediction_horizon = 2  # Number of steps ahead to predict
    X, Y = create_dataset(calories_normalized, time_step, prediction_horizon)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # Add feature dimension
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

    # Define LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    # Initialize model, loss function, and optimizer
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Predict future values
    model.eval()
    with torch.no_grad():
        future_steps = 5  # Number of days you want to predict into the future
        predictions = []
        input_seq = torch.tensor(calories_normalized[-time_step:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for _ in range(future_steps):
            pred = model(input_seq)
            predictions.append(pred.item())
            input_seq = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(-1)), dim=1)

    # Inverse transform the predictions
    predictions = np.array(predictions).flatten() * calories_std + calories_mean

    # Generate future dates
    last_date = data['date'].max()
    future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, future_steps + 1)]

    # Create a DataFrame for plotting
    plot_dates = np.concatenate([data['date'], future_dates])
    plot_calories = np.concatenate([calories, predictions])

    # Plot the results

    model.eval()
    with torch.no_grad():
        predictionsp = model(X).detach().numpy()

    # Inverse transform the predictions
    predictionsp = predictionsp.flatten() * calories_std + calories_mean

    # Generate dates for predictions
    predicted_dates = data['date'][
                      time_step + prediction_horizon - 1: len(predictionsp) + time_step + prediction_horizon + 1]

    # Create a DataFrame for plotting
    plot_datesp = np.concatenate([data['date'][10:time_step + prediction_horizon + 1], predicted_dates])
    plot_caloriesp = np.concatenate([calories[10:time_step + prediction_horizon + 1], predictionsp])

    # Plot the results

    plt.figure(figsize=(12, 6))
    plt.plot(data['date'][len(data) - 10:], calories[len(data) - 10:], label='Original Calorie Intake')
    plt.plot(future_dates, predictions, label='Predicted Calorie Intake', linestyle='--')
    plt.plot(plot_datesp[len(plot_datesp) - 10:], plot_caloriesp[len(plot_datesp) - 10:],
             label='Predicted Calorie Intake', linestyle='--')
    # plt.scatter(future_dates, predictions, color='red', zorder=5)
    # for i, txt in enumerate(predictions):
    #     plt.text(future_dates[i], predictions[i], f'{txt:.2f}', verticalalignment='bottom',
    #              horizontalalignment='left')
    plt.xlabel('Date')
    plt.ylabel('Calories')
    plt.title('Daily Calorie Intake and Forecast')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Send the plot as a response
    return send_file(img, mimetype='image/png', as_attachment=True, download_name='daily_calorie_plot.png')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('love')
    app.run()






