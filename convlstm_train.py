import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from convlstm import ConvLSTM
import numpy as np

cities = ["XA", "CD"]
data_paths = {
    "XA": {
        "speed": "../data/speed_XA.npy",
        "inflow": "../data/inflow_XA.npy",
        "demand": "../data/demand_XA.npy",
    },
    "CD": {
        "speed": "../data/speed_CD.npy",
        "inflow": "../data/inflow_CD.npy",
        "demand": "../data/demand_CD.npy",
    },
}

model_folder = "./models"
os.makedirs(model_folder, exist_ok=True)

def process_small(speed, demand, inflow):
    demand_threshold = torch.quantile(demand, 0.9)
    inflow_threshold = torch.quantile(inflow, 0.9)

    speed = torch.clamp(speed, max=140)
    demand = torch.clamp(demand, max=demand_threshold)
    inflow = torch.clamp(inflow, max=inflow_threshold)

    def normalize(data):
        min_val = data.min()
        max_val = data.max()
        if max_val - min_val == 0:
            return torch.zeros_like(data)
        return 2 * (data - min_val) / (max_val - min_val) - 1

    normalized_speed = normalize(speed).float()
    normalized_demand = normalize(demand).float()
    normalized_inflow = normalize(inflow).float()

    return normalized_speed, normalized_demand, normalized_inflow

def train(city, data_type, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {data_type} model for {city} on {device}")

    model = ConvLSTM(
        input_dim=1,
        hidden_dim=16,
        kernel_size=(3, 3),
        num_layers=3,
        batch_first=True,
        predict_channel=1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500, 1000], gamma=0.1)

    batch_size, seq_len, num_regions, height, width = data.size()

    num_epochs = 200
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for region_idx in range(num_regions):
            region_data = data[:, :, region_idx, :, :].unsqueeze(2)
            dataloader = DataLoader(torch.utils.data.TensorDataset(region_data), batch_size=64, shuffle=True)

            for batch_data in dataloader:
                batch_data = batch_data[0].to(device)
                input_seq = batch_data[:, :-2, :, :, :]
                target = batch_data[:, -2, :, :, :]

                optimizer.zero_grad()
                layer_output_list, last_state_list, output = model(input_seq)

                loss = torch.nn.functional.mse_loss(output, target)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

        scheduler.step()

        print(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss / (num_regions * len(dataloader)):.6f}")

        if epoch % 200 == 0 or epoch == num_epochs:
            model_path = os.path.join(model_folder, f"{city}_{data_type}_epoch{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            with torch.no_grad():
                for region_idx in range(num_regions):
                    region_data = data[:, :, region_idx, :, :].unsqueeze(2).to(device)
                    layer_output_list, last_state_list, output = model(region_data)

                    print(f"Epoch {epoch}, Region {region_idx + 1}:")
                    print(f"  layer_output_list[-1].shape: {layer_output_list[-1].shape}")
                    print(f"  last_state_list[-1][0].shape: {last_state_list[-1][0].shape}")
                    print(f"  output.shape: {output.shape}")

def main():
    for city in cities:
        speed = torch.tensor(np.load(data_paths[city]["speed"]))
        inflow = torch.tensor(np.load(data_paths[city]["inflow"]))
        demand = torch.tensor(np.load(data_paths[city]["demand"]))

        speed, demand, inflow= process_small(speed, demand, inflow)

        for data_type, data in zip(["speed", "inflow", "demand"], [speed, inflow, demand]):
            train(city, data_type, data)

if __name__ == "__main__":
    main()
