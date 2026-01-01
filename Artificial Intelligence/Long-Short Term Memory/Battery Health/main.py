
# main.py

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.
#TODO AIC: Add input control.

#! PW: Possibly wrong.

from typing import Tuple

from torch.utils.data import DataLoader

from data_preprocess import (
    ConfigDataManager,
    DataManager,
)
from graph import (
    ConfigHistoryPlotter,
    HistoryPlotter
)
from lstm import (
    ConfigLSTM,
    LSTM
)
from train import (
    ConfigTrainer,
    Trainer
)

if __name__ == "__main__":
    # 1. Define the configuration and instance for data preprocessing.
    config_data_manager = ConfigDataManager(
        stride = 1,
        mat_file_name = "B0005.mat",
        battery_id = "B0005",
        interpolation_length = 500,
        train_split = 0.65,
        val_split = 0.20,
        batch_size = 32
    )

    data_manager = DataManager(config_data_manager=config_data_manager)

    # 2. Preprocess the data and return the dataloaders.
    data_manager.prepare_data()

    dataloaders: Tuple[DataLoader, DataLoader, DataLoader] = data_manager.get_loaders()

    # 3. Define the configuration and instance for LSTM Architecture.
    config_lstm = ConfigLSTM(
        input_size = 3,
        hidden_size = 32,
        num_layers = 1,
        output_size = 1,
        bidirectional = False,
        dropout_p = 0.2
    )

    model = LSTM(config_lstm=config_lstm)

    # 4. Define the configuration and instance for training.
    config_trainer = ConfigTrainer(
        dataloaders = dataloaders,
        learning_rate = 1e-3,
        num_epochs = 10000,
        max_norm = 1.0,
        print_freq = 500,
        val_freq = 500,
        save_checkpoint_freq = 500,
        model_save_path = "C:/Users/Besitzer/Desktop/Python/AI Projects/Long-Short Time Memory/Battery Health/Trials",
        saved_checkpoint = None,
        show_graph = True
    )

    trainer = Trainer(
        config_trainer = config_trainer,
        model = model
    )

    # 5. Start the training.
    avg_loss_history, rmse_loss_history, r2_loss_history, mae_loss_history, train_time, trial_folder_path = trainer.fit()

    # Test the model after training.
    # TODO

    # 6. Define the configuration and instance for the graphs.
    config_history_plotter = ConfigHistoryPlotter(train_time = train_time)

    history_plotter = HistoryPlotter(config_history_plotter=config_history_plotter)

    # 7. Plot the graphs.
    history_plotter.plot_history(
        histories = (avg_loss_history, rmse_loss_history, r2_loss_history, mae_loss_history),
        plot_colors = ("red", "blue"),
        plot_linestyles = ("--", "--"),
        trial_folder_path = trial_folder_path,
        show_graph = True
    )

