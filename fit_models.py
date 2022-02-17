#!/usr/bin/env python

"""
This script is intended for statistical seasonal forecasting of temperatures in
different regions in Europe. It  uses random sampling and LASSO regression
to build an ensemble of models which are saved in a pickle object to be analyzed later.

All input data is preprocessed inside the READ_DEFINE_AND_PROCESS_EVERYTHING routine. 
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

import functions as fcts


def main(y_var, area, exp, src, in_dir: Path, out_dir: Path):
    # Read and process all input data, define variables etc.
    data = fcts.READ_DEFINE_AND_PROCESS_EVERYTHING(in_dir, y_var, area, exp, src)

    print('Fitting models for', data['basename'], data['experiment'])

    # Create the output data table
    columns = ['Season', 'Fitted model', 'Optimal predictors', 'Indexes of optimal predictors',
               'N of optimal predictors', 'In-sample ACC', 'Model weight']

    models_out = pd.DataFrame(columns=columns)

    # Fit models for each season separately
    for l, ssn in enumerate(data['seasons']):
        print('Training years:', data['trn_yrs'])
        n_estimators = data['n_smpls']

        # Indexes for selecting data from the training period
        trn_idx = fcts.bool_index_to_int_index(
            np.isin(data['Y']['time.season'], ssn) & np.isin(data['Y']['time.year'], data['trn_yrs']))

        # Fit LassoLarsCV models using the handy BaggingRegressor meta-estimator
        cv = KFold(n_splits=5, shuffle=True)
        base_estimator = LassoLarsCV(eps=2e-10, max_iter=200, cv=cv, n_jobs=1, normalize=False)
        ensemble = fcts.bagging_metaestimator(data['X'].values[trn_idx], data['Y'][data['y_var']].values[trn_idx],
                                              data['vrbl_names'],
                                              data['n_smpls'], data['p_smpl'], data['p_feat'], data['n_jobs'],
                                              base_estimator)

        # Append the models to the output table, including also the season information
        for i, mdl in enumerate(ensemble.estimators_[:n_estimators]):
            feature_idxs = ensemble.estimators_features_[i]
            posit_features = np.abs(mdl.coef_) > 0
            feature_names = list(data['vrbl_names'][feature_idxs][posit_features])
            n_features = len(feature_names)
            fcs = mdl.predict(data['X'].values[trn_idx][:, feature_idxs])
            obs = data['Y'][data['y_var']].values[trn_idx]
            train_period_acc = fcts.calc_corr(fcs, obs)
            df = pd.DataFrame([[ssn, mdl, feature_names, feature_idxs, n_features, train_period_acc, -99]],
                              columns=columns)
            models_out = pd.concat([models_out, df])

        # Try weighting models based on their validation scores
        ssn_idx = models_out['Season'] == ssn
        weights = MinMaxScaler().fit_transform(models_out[ssn_idx]['In-sample ACC'].values.reshape(-1, 1))
        models_out['Model weight'].loc[ssn_idx] = weights.squeeze()

        print('Completed', len(ensemble), 'succesful fittings for', ssn, data['y_var'], data['y_area'])

    # Save results into a Pandas pickle object
    out_dir.mkdir(parents=True, exist_ok=True)
    models_out.to_pickle(out_dir / (data['basename'] + '.pkl'))


if __name__ == "__main__":
    # Target variable
    y_var = 'T2M'

    # Target domains
    # areas=('europe' 'scandi' 'easeur' 'westeu' 'meditr')
    area = 'europe'

    # Experiment
    # Possible values: 'CONTROL', 'INCLPERSIS', 'SKIPQMAP', 'DETREND', 'NO_LAGS', 'CUTFIRSTYRS', 'SINGLE', 'WEIGHTING', 'FOLLAND'
    exp = 'CONTROL'

    # Source for predictor data
    # Possible values: 'ERA-20C', '20CRv2c'
    src = 'ERA-20C'

    # Directories
    basedir = Path(__file__).parent
    in_dir = Path(__file__).parent  # Dónde estan los .nc que me he descargado
    out_dir = Path(__file__).parent / 'results'

    main(y_var, area, exp, src, in_dir, out_dir)
