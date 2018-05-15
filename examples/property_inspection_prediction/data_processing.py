#!/usr/bin/env python

import os
import numpy as np
import pandas as pd


if __name__ == '__main__':

    seed = 0
    test_pct = 0.3

    np.random.seed(seed)

    try:
        working_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

    except NameError:
        working_dir = os.path.join(
            os.getcwd(),
            'examples',
            'property_inspection_prediction'
        )

    data = pd.read_csv(
        os.path.join(
            working_dir,
            'train.csv'
        ),
        header=None
    )

    cat_cols = data.columns[data.dtypes != 'int64']
    for col in cat_cols:
        unique_chars = np.unique(data[col])
        code = -1
        for uc in unique_chars:
            code += 1
            data.loc[data[col] == uc, col] = code

    row_idx = data.index.values
    np.random.shuffle(row_idx)
    train_row_idx = row_idx[int(test_pct * len(data)):]
    test_row_idx = row_idx[:int(test_pct * len(data))]

    train_data = data.iloc[train_row_idx, :]
    test_data = data.iloc[test_row_idx, :]

    train_data.to_csv(
        os.path.join(
            working_dir,
            'poisson.train'
        ),
        header=False,
        index=False
    )
    test_data.to_csv(
        os.path.join(
            working_dir,
            'poisson.test'
        ),
        header=False,
        index=False
    )

    pd.Series(cat_cols).to_csv(
        os.path.join(
            working_dir,
            'cat_col_idx.test'
        ),
        index=False
    )