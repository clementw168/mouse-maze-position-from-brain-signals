import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from dataset import get_dataloader
from models import SpikeEmbeddingModel, WaveformModel


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        groups, pos, length, gathered_spikes, is_not_padding = batch
        groups = groups.to(device)
        gathered_spikes = gathered_spikes.to(device)
        pos = pos.to(device)
        is_not_padding = is_not_padding.to(device)

        outputs = model(groups, length, gathered_spikes, is_not_padding)
        loss = loss_fn(outputs, pos)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    for batch in tqdm(dataloader):
        groups, pos, length, gathered_spikes, is_not_padding = batch
        groups = groups.to(device)
        gathered_spikes = gathered_spikes.to(device)
        pos = pos.to(device)
        is_not_padding = is_not_padding.to(device)

        outputs = model(groups, length, gathered_spikes, is_not_padding)
        loss = loss_fn(outputs, pos)
        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    import argparse

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_id", type=str, default="M1182_PAG")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=108)
    parser.add_argument("--split_type", type=str, default="temporal")
    parser.add_argument("--loss_file", type=str, default=f"{timestamp}.png")
    parser.add_argument("--weights_file", type=str, default=f"{timestamp}.pth")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer_params = {
        "optimizer": torch.optim.Adam,
        "lr": 0.002,
        "loss_fn": torch.nn.MSELoss(),
        "epochs": args.epochs,
        "batch_size": 256,
        "loss_file": f"losses/{args.loss_file}",
        "weights_file": f"weights/{args.weights_file}",
    }

    os.makedirs("losses", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    (
        train_loader,
        test_loader,
        max_channels,
    ) = get_dataloader(
        args.mouse_id,
        args.stride,
        args.window_size,
        batch_size=trainer_params["batch_size"],
        split_type=args.split_type,
    )

    ## Base params

    # model_class = SpikeEmbeddingModel
    # params = {
    #     "max_groups": max_channels,
    #     "hidden_size": 64,
    #     "num_conv_layers": 3,
    #     "num_fc_layers": 2,
    #     "kernel_size": 5,
    #     "stride": 3,
    # }

    # model_class = WaveformModel
    # params = {
    #     "max_groups": max_channels,
    #     "hidden_size": 64,
    #     "num_conv_layers": 3,
    #     "num_fc_layers": 2,
    #     "kernel_size": 5,
    #     "stride": 2,
    # }

    ## Manual tuning

    model_class = WaveformModel
    params = {
        "max_groups": max_channels,
        "hidden_size": 64,
        "num_conv_layers": 3,
        "num_fc_layers": 2,
        "kernel_size": 5,
        "stride": 2,
    }

    model = model_class(**params).to(device)

    optimizer = trainer_params["optimizer"](model.parameters(), lr=trainer_params["lr"])

    loss_fn = torch.nn.MSELoss()

    train_losses = []
    test_losses = []
    for epoch in range(trainer_params["epochs"]):
        print(f"Epoch {epoch}")
        train_loss = train_loop(train_loader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss}")
        test_loss = test_loop(test_loader, model, loss_fn, device)
        test_losses.append(test_loss)
        print(f"Test Loss: {test_loss}\n")

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(trainer_params["loss_file"])
    plt.close()

    torch.save(model.state_dict(), trainer_params["weights_file"])

    pandas_row = {
        "mouse_id": args.mouse_id,
        "stride": args.stride,
        "window_size": args.window_size,
        "split_type": args.split_type,
        "loss_file": trainer_params["loss_file"],
        "weights_file": trainer_params["weights_file"],
        "train_loss": train_losses,
        "test_loss": test_losses,
        "best_train_loss": min(train_losses),
        "best_test_loss": min(test_losses),
    }

    csv_path = "training_log.csv"
    df_row = pd.DataFrame([pandas_row])
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=write_header, index=False)
