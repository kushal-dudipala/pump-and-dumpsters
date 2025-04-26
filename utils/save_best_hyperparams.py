import json
import os
from datetime import datetime

def save_best_hyperparams(model_name, best_params, best_score, save_dir="saved_hyperparams"):
    """
    Saves the best hyperparameters and score to a JSON file.

    Parameters
    ----------
    model_name : str
        The name of the model ('lstm', 'cnn', 'hybrid').
    best_params : tuple
        The best hyperparameters found.
    best_score : float
        The best F1 score achieved.
    save_dir : str
        Directory where the JSON file will be saved (default: 'saved_hyperparams').
    """
    os.makedirs(save_dir, exist_ok=True)

    save_data = {
        "model": model_name,
        "best_params": best_params,
        "best_score": best_score,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save to JSON
    filename = f"{model_name}_best_hyperparams.json"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=4)

    print(f"\nBest hyperparameters saved to: {filepath}")
