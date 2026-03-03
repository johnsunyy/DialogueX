"""
training/tuner.py

Optuna-based hyperparameter search for ASL MLP.
Maximises validation macro F1 across 30 trials (configurable).

After study completion:
  - Prints best hyperparameters
  - Retrains best model on train+val data
  - Saves to best_model.pt

Usage:
    python -m training.tuner [--trials 30] [--epochs 100] [--batch_size 512]
"""

import os
import sys
import pickle
import argparse
import numpy as np

import optuna
from optuna.samplers import TPESampler
import torch

ROOT_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
LOGS_DIR      = os.path.join(ROOT_DIR, 'logs')
BEST_MODEL    = os.path.join(ROOT_DIR, 'best_model.pt')

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(processed_dir: str):
    def _load(name):
        path = os.path.join(processed_dir, f'{name}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run preprocessing first.")
        return np.load(path)

    return (
        _load('X_train'), _load('X_val'), _load('X_test'),
        _load('y_train'), _load('y_val'), _load('y_test'),
    )


def make_objective(X_train, y_train, X_val, y_val, epochs: int, batch_size: int):
    """Factory returning the Optuna objective function."""

    from training.model import build_model_from_trial
    from training.train import train_model

    def objective(trial):
        # Build model from trial parameters
        model = build_model_from_trial(trial, input_dim=63, num_classes=26)

        # Suggest training hyperparameters
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
        weight_decay   = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        val_f1 = train_model(
            model,
            X_train, y_train,
            X_val,   y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=8,          # shorter patience during search
            optimizer_name=optimizer_name,
            save_path=None,      # don't save every trial
            log_path=None,
            verbose=False,       # silent during search
        )

        return val_f1

    return objective


def run_tuning(
    processed_dir: str,
    n_trials: int = 30,
    epochs: int = 100,
    batch_size: int = 512,
):
    print(f"\n{'='*60}")
    print(f"  Optuna Hyperparameter Search")
    print(f"  Trials      : {n_trials}")
    print(f"  Max epochs  : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"{'='*60}\n")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(processed_dir)

    # Combine train+val for final retraining
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='asl_landmark_mlp',
    )

    objective = make_objective(X_train, y_train, X_val, y_val, epochs, batch_size)

    print(f"  Running {n_trials} trials... (progress printed every 5)\n")
    completed = [0]

    def callback(study, trial):
        completed[0] += 1
        if completed[0] % 5 == 0 or completed[0] == 1:
            print(f"  Trial {completed[0]:>3}/{n_trials}  "
                  f"best val F1 so far: {study.best_value:.4f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    print(f"\n{'='*60}")
    print(f"  Search Complete!")
    print(f"  Best val F1  : {study.best_value:.4f}")
    print(f"  Best params  :")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    # ── Retrain best model on train+val ──────────────────────────
    from training.model import build_model_from_trial, ASLClassifier
    from training.train import train_model

    print("  Retraining best model on combined train+val data...")

    best_trial = study.best_trial
    best_model = build_model_from_trial(best_trial, input_dim=63, num_classes=26)

    # Load best params
    lr            = best_trial.params['lr']
    optimizer_name = best_trial.params['optimizer']
    weight_decay  = best_trial.params.get('weight_decay', 1e-4)

    # Use original val as a proxy for early stopping during final retraining
    final_f1 = train_model(
        best_model,
        X_trainval, y_trainval,
        X_val,      y_val,        # still use val for early stop
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=15,
        optimizer_name=optimizer_name,
        save_path=BEST_MODEL,
        log_path=os.path.join(LOGS_DIR, 'tuned_training_log.json'),
        verbose=True,
    )

    # ── Evaluate on test set ─────────────────────────────────────
    print("\n  Evaluating best model on held-out TEST set...")
    from evaluation.metrics import evaluate_and_report

    # Load best checkpoint
    ckpt = torch.load(BEST_MODEL, map_location='cpu')
    best_model.load_state_dict(ckpt['model_state_dict'])

    le_path = os.path.join(ROOT_DIR, 'label_encoder.pkl')
    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    evaluate_and_report(best_model, X_test, y_test, le.classes_)
    print(f"\n  Tuned model saved to: {BEST_MODEL}")


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter tuning for ASL MLP')
    parser.add_argument('--trials',        type=int, default=30)
    parser.add_argument('--epochs',        type=int, default=100)
    parser.add_argument('--batch_size',    type=int, default=512)
    parser.add_argument('--processed_dir', default=PROCESSED_DIR)
    args = parser.parse_args()

    run_tuning(
        processed_dir=os.path.abspath(args.processed_dir),
        n_trials=args.trials,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
