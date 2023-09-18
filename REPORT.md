    
cfg = {
    'dev_train_len': 25*10**3,
    'dev_validation_len': 5*10**3,
    'learning_rate': 0.001,
    'epochs': 100,
    'embedding_dim': 50,
    'batch_size': 32,
    'dropout': 0,
    'optimizer': 'Adam',
    'num_layers': 2
}

train_len: 3 * dev_train_len
validation = 3 * validation_len

cfg['hidden_dim'] = cfg['embedding_dim']

    

    delta = [Trained, [3, 0, 0], [1, 2, 3], [0, 0, 3], [1, 1, 1]] 

#### Learning the delta parameter

**Confusion Matrix**

    [
        [1588   93  108  111]
        [  98 1707   31   64]
        [ 116   49 1417  318]
        [ 124   47  251 1478]
    ]

**Classification Report**

                precision    recall  f1-score   support

            0     0.8245    0.8358    0.8301      1900
            1     0.9003    0.8984    0.8994      1900
            2     0.7842    0.7458    0.7645      1900
            3     0.7499    0.7779    0.7636      1900

    accuracy                          0.8145      7600
    macro avg     0.8147    0.8145    0.8144      7600
    weighted avg  0.8147    0.8145    0.8144      7600

**Test loss**

Test loss: 0.5201968483063353

#### Delta parameter: [3, 0, 0]

**Confusion Matrix**

    [
        [1636   68  115   81]
        [  37 1806   22   35]
        [ 108   17 1537  238]
        [  89   20  224 1567]
    ]

**Classification Report**

                precision    recall  f1-score   support

            0     0.8749    0.8611    0.8679      1900
            1     0.9451    0.9505    0.9478      1900
            2     0.8098    0.8089    0.8094      1900
            3     0.8157    0.8247    0.8202      1900

        accuracy                      0.8613      7600
    macro avg     0.8614    0.8613    0.8613      7600
    weighted avg  0.8614    0.8613    0.8613      7600

**Test loss**

0.4370632403287567

#### Delta parameter: [1, 2, 3]

**Confusion Matrix**

    [
        [1485  165  145  105]
        [  76 1699   43   82]
        [  88   52 1402  358]
        [ 128  101  312 1359]
    ]

**Classification Report**

                precision    recall  f1-score   support

            0     0.8357    0.7816    0.8077      1900
            1     0.8423    0.8942    0.8675      1900
            2     0.7371    0.7379    0.7375      1900
            3     0.7138    0.7153    0.7145      1900

        accuracy                      0.7822      7600
    macro avg     0.7822    0.7822    0.7818      7600
    weighted avg  0.7822    0.7822    0.7818      7600

**Test loss**

0.5831848365419051

#### Delta parameter: [0, 0, 3]

**Confusion Matrix**

    [
        [1359  232  132  177]
        [ 216 1535   32  117]
        [ 288   73  973  566]
        [ 248  135  268 1249]
    ]

**Classification Report**

                precision    recall  f1-score   support

            0     0.6438    0.7153    0.6776      1900
            1     0.7772    0.8079    0.7923      1900
            2     0.6925    0.5121    0.5888      1900
            3     0.5922    0.6574    0.6231      1900

        accuracy                      0.6732      7600
    macro avg     0.6764    0.6732    0.6704      7600
    weighted avg  0.6764    0.6732    0.6704      7600

**Test loss**

0.7946344824398265

#### Delta parameter: [1, 1, 1]

**Confusion Matrix**

    [
        [1574  126  125   75]
        [ 103 1705   42   50]
        [ 131   55 1459  255]
        [ 180   90  324 1306]
    ]

**Classification Report**

                precision    recall  f1-score   support

            0     0.7918    0.8284    0.8097      1900
            1     0.8629    0.8974    0.8798      1900
            2     0.7482    0.7679    0.7579      1900
            3     0.7746    0.6874    0.7284      1900

        accuracy                      0.7953      7600
    macro avg     0.7944    0.7953    0.7939      7600
    weighted avg  0.7944    0.7953    0.7939      7600

**Test loss**

0.5559738839373869