import wandb

config = {
    'method': 'random',
    'name': 'ELMO',
    'metric': {
        'goal': 'minimize',
        'name': 'loss'
    },
    # 'parameters': {
    #     # 'train_len': {'value': 3*10**4},
    #     'dev_train_len': {'value': 25*10**3},
    #     'dev_validation_len': {'value': 5*10**3},
    #     # 'validation_len': {'value': 10**4},
    #     # 'test_len': {'value': 14 * 10**3},
    #     'learning_rate': {'value': 0.001},
    #     'epochs': {'value': 100},
    #     'embedding_dim': {'values': [100, 200]},
    #     'hidden_dim': {'values': [300, 500, 750, 1000]},
    #     'dropout': {'values': [0, 0.15, 0.30]},
    #     'optimizer': {'value': 'Adam'},
    #     'num_layers': {'value': 2}
    # }
    'parameters': {
        # 'train_len': {'value': 3*10**4},
        'dev_train_len': {'value': 5*10**3},
        'dev_validation_len': {'value': 1*10**3},
        # 'validation_len': {'value': 10**4},
        # 'test_len': {'value': 14 * 10**3},
        'learning_rate': {'value': 0.001},
        'epochs': {'value': 100},
        'embedding_dim': {'value': 50},
        'hidden_dim': {'value': 100},
        'dropout': {'value': 0},
        'optimizer': {'value': 'Adam'},
        'num_layers': {'value': 2}
    }
}