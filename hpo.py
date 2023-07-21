import os
import pickle
import click
import mlflow
import optuna
import numpy as np

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# from preprocess_data import read_dataframe, fill_dataframe

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("quality-of-air-forest-hyperopt")


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./dataset",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=1000,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "valid.pkl"))

    # df_test = read_dataframe(
    #     os.path.join('data', "Test.csv"), train=False
    # )

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 1000, 1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
            'random_state': 42,
            'n_jobs': -1
        }

        with mlflow.start_run():
            
            mlflow.set_tag("model", "RandomForest")
            mlflow.set_tag("developer", "Moses Daudu")
            mlflow.log_params(params)

            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            mlflow.sklearn.log_model(rf, 'rf-model')

            y_pred = rf.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mlflow.log_metric('rmse', rmse)
            print('The root meansquared error is:', rmse)

        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':
    run_optimization()
