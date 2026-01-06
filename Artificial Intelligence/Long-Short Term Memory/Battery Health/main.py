
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
from graph import Plotter
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
        data_directory = "CHANGE THIS PART",
        train_battery_ids = ["B0005", "B0006"],
        val_battery_ids = ["B0007"],
        test_battery_ids = ["B0018"],
        interpolation_length = 500,
        batch_size = 32
    )

    data_manager = DataManager(config_data_manager=config_data_manager)

    # 2. Preprocess the data and return the dataloaders.
    test_scaling_factors = data_manager.prepare_data()

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
        mode = "min",
        factor = 5e-1,
        patience = 5,
        min_lr = 1e-5,
        num_epochs = 10000,
        max_norm = 1.0,
        print_freq = 1000,
        val_freq = 1000,
        save_checkpoint_freq = 2000,
        model_save_path = "CHANGE THIS PART",
        saved_checkpoint = None,
        show_graph = True
    )

    trainer = Trainer(
        config_trainer = config_trainer,
        model = model
    )

    # 5. Start the training.
    train_mse_history, train_rmse_history, train_r2_history, train_mae_history, train_time, trial_folder_path = trainer.fit()

    # Test the model after training.
    test_results = trainer.test()

    # 6. Define the instance for the graphs.
    plotter = Plotter()

    # 7. Plot the graphs.
    plotter.plot_training_results(
        train_time = train_time,
        histories = (train_mse_history, train_rmse_history, train_r2_history, train_mae_history),
        colors = ("red", "blue"),
        linestyles = ("--", "--"),
        trial_folder_path = trial_folder_path,
        show_graph = True
    )

    plotter.plot_test_results(
        scaling_factors = test_scaling_factors,
        data = test_results,
        colors = ("red", "blue"),
        linestyles = ("-", "-"),
        trial_folder_path = trial_folder_path,
        show_graph = True
    )

