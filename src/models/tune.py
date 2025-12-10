import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

def tune_model(X, y):
    def objective(trail):
        params={
            'n_estimators': trail.suggest_int('n_estimators', 300, 800),
            'learning_rate': trail.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trail.suggest_int('max_depth', 3, 10),
            'subsample': trail.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trail.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trail.suggest_int('min_child_weight', 1, 10),
            'gamma': trail.suggest_float('gamma', 0, 5),
            'reg_alpha': trail.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trail.suggest_float('reg_lambda', 0, 5),
            'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params)
        score = cross_val_score(model, X, y, cv=3, scoring='recall').mean()
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters:", study.best_params)
    return study.best_params