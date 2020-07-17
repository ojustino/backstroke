#!/usr/bin/python3
from datetime import datetime, timedelta
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
import numpy as np
import pandas as pd
import requests
import time
import warnings


class PortfolioMaker:
    # INCORPORATE traitlets to do the validation?
    '''
    Create a portfolio of assets to be fed into a simulation. The centerpiece of
    this class is self.assets, a dictionary containing the tickers to be used in
    the simulation and their associated information.

    Portfolios follow the core/satellite model -- a static core portion whose
    assets are consistently rebalanced to the same target weights, and a more
    dynamic satellite portion that can include a riskier "in-market" asset
    and a safer "out-of-market" asset.

    self.assets can optionally keep track of a separate portfolio of one or more
    benchmark assets that follow the same rebalancing schedule as the core
    portion of the main portfolio. Benchmark assets can be used to help make
    decisions in Strategies or just to provide a baseline against which to
    measure the main core/satellite portfolio's success.

    The self.add_ticker() method is the primary mechanism for adding information
    to the assets dictionary.

    Arguments
    ---------

    sat_frac : float, required
        A number between 0 and 1 (inclusive) that dictates what (decimal)
        fraction of the main portfolio should be allocated to the satellite
        portion, with the core taking up the other 1 - `sat_frac` fraction.

    relative_core_frac : boolean, optional
        A convenience option that, when True, automatically adjusts individual
        core assets' fractions when `sat_frac` is changed. [default: True]
    '''
    def __init__(self, sat_frac, relative_core_frac=True):
        # download ticker data
        self._valid_tix = self._fetch_ticker_info()

        # set up initial class attributes
        self.assets = {}
        self.tick_info = pd.DataFrame()
        # will eventually contain rows of assets selected from valid_tix

        # set __init__'s arguments as properties so changes are tracked
        self._sat_frac = self._validate_fraction(sat_frac)
        self._relative_core_frac = relative_core_frac

    @property
    def sat_frac(self):
        '''
        Controls how much of the main portfolio is allotted to 'satellite'-
        labeled tickers in self.assets (as opposed to 'core').
        '''
        return self._sat_frac

    @sat_frac.setter
    def sat_frac(self, val):
        '''
        When self.sat_frac changes and self.relative_core_frac is True, core
        asset fractions are readjusted relative to the new value. Otherwise, the
        change takes place without any further action.
        '''
        # ensure that new value is acceptable
        val = self._validate_fraction(val)

        if self.relative_core_frac and val != self._sat_frac:
            old_sat_frac = self._sat_frac

            for info in self.assets.values():
                if info['label'] == 'core':
                    old_frac = info['fraction']
                    info['fraction'] = np.round(old_frac / (1 - old_sat_frac)
                                                * (1 - val), 6)
                    # np.testing.approx_equal? Decimal (to treat as exact number instead of float)?
            print('core fractions adjusted relative to new `sat_frac` value')

        self._sat_frac = val

    @property
    def relative_core_frac(self):
        '''
        Controls whether 'core' asset fractions are adjusted based on
        self.sat_only. If True, each 'core' asset fraction is multiplied by
        (1 - self.sat_frac) -- the core's fraction of the portfolio.
        '''
        return self._relative_core_frac

    @relative_core_frac.setter
    def relative_core_frac(self, val):
        '''
        When self.relative_core_frac changes, core asset fractions are either
        adjusted relative to self.sat_frac (False to True) or to their
        originally entered values (True to False).
        '''
        # ensure that new value is acceptable
        val = self._validate_relative(val)

        # if was previously False, proportionally downsize core asset weights
        if self._relative_core_frac == False and val == True:
            for info in self.assets.values():
                if info['label'] == 'core':
                    old_frac = info['fraction']
                    info['fraction'] = np.round(old_frac * (1-self.sat_frac), 6)
            print('core fractions reset relative to `sat_frac` value')

        # if was previously True, return core asset weights to original values
        # entered when they were called with self.add_ticker()
        elif self._relative_core_frac == True and val == False:
            for info in self.assets.values():
                if info['label'] == 'core':
                    old_frac = info['fraction']
                    info['fraction'] = np.round(old_frac / (1-self.sat_frac), 6)
            print('core fractions reset to their originally entered values')

        self._relative_core_frac = val

    def _fetch_ticker_info(self):
        '''
        Before we begin, we use requests to download a zipped CSV from Tiingo
        that lists valid tickers with their exchanges, currencies, and start/
        end dates.

        For whatever reason, it's not as simple as calling
        pd.read_csv(tick_url), but a technique I found in
        github.com/hydrosquall/tiingo-python/blob/master/tiingo/api does work.
        '''
        TICK_URL = ('https://apimedia.tiingo.com/docs/'
                    'tiingo/daily/supported_tickers.zip')

        # download the zipped csv data and convert it to be usable by pandas
        _valid_zip = requests.get(TICK_URL)
        _valid_zip_data = ZipFile(BytesIO(_valid_zip.content))
        _valid_zip_byte = BytesIO(_valid_zip_data.read('supported_tickers.csv'))
        _valid_zip_buff = TextIOWrapper(_valid_zip_byte)

        # next, make it a DataFrame...
        _valid_tix = pd.read_csv(_valid_zip_buff)

        # ...and only keep assets traded in USD and have valid start/end dates
        # (some have null 'exchange' values; does it matter?)
        valid_tix = _valid_tix[(_valid_tix.loc[:,'priceCurrency'] == 'USD')
                               & (_valid_tix.loc[:,'startDate'].notnull())]

        # to view specific tickers:
        # valid_tix[valid_tix['ticker'].isin(['SCHG', 'SCHM', 'EFG',
        #                                    'BIV', 'LQD', 'ACES'])]

        return valid_tix

    def _validate_fraction(self, fraction):
        '''
        Ensures the proposed self.sat_frac or core/benchmark fraction value is a
        number in the proper range.

        Arguments
        ---------

        fraction : int or float, required
            The proposed fraction of the portfolio.
        '''
        if (not (isinstance(fraction, int) or isinstance(fraction, float))
            or (not 0 <= fraction <= 1)):
            raise ValueError('fraction must be a number from 0 to 1')
        return fraction

    def _validate_label(self, label, in_market):
        '''
        Ensures the proposed label is in the list of permitted names. For
        satellite assets, also checks whether a market position was specified.

        Arguments
        ---------

        label : str, required
            The proposed asset's type -- core, satellite, or benchmark.

        in_market : boolean, required
            Only needed if `label` == satellite. If True, this asset will be
            in-market. If False, it will be the out-of-market asset.
        '''
        if label is None:
            label = 'core'
        elif label == 'satellite':
            if len(self._get_label_tickers(label)) == 2:
                raise ValueError('For now, there can only be up to two '
                                 'satellite assets. Please remove an existing '
                                 'entry if you prefer to use this one.')
            self._validate_in_market(in_market)
        elif label not in {'core', 'satellite', 'benchmark'}:
            raise ValueError("Valid `label` options are 'core', 'satellite', "
                             "and 'benchmark'.")
        return label

    def _validate_in_market(self, in_market):
        '''
        Ensures the proposed in_market value has been set properly for a
        corresponding satellite asset.

        Arguments
        ---------

        in_market : boolean, required
            If True, this asset will be in-market. If False, it will be the
            out-of-market asset.
        '''
        if not isinstance(in_market, bool):
            raise ValueError('Satellite tickers must specify whether they '
                             'are the in-market or out-market asset. '
                             'Please set `in_market` to True or False.')
        return in_market

    def _validate_relative(self, rel_core_frac):
        '''
        Ensures the proposed self.relative_core_frac value is a boolean.

        Arguments
        ---------

        rel_core_frac : boolean, required
            If True, automatically adjusts individual core assets' fractions
            when self.sat_frac is changed.
        '''
        if not isinstance(rel_core_frac, bool):
            raise TypeError('`relative_core_frac` must be a boolean.')

        return rel_core_frac

    def _validate_ticker(self, ticker):
        '''
        Ensures the proposed ticker is part of self._valid_tix, the DataFrame of
        tickers supported by Tiingo. Also checks whether it already exists in
        self.assets.

        Arguments
        ---------

        ticker : str, required
            The symbol of the proposed asset.
        '''
        if ticker not in self._valid_tix['ticker'].values:
            raise ValueError('{ticker} is absent from the Tiingo list of '
                             'supported assets. Try another?')

        if ticker in self.assets.keys():
            raise ValueError('`ticker` value is already a key in `assets`. '
                             'To replace the entry, first use remove_ticker() '
                             'on your proposed `ticker` value.')

        return ticker

    def _validate_shares(self, shares, label):
        '''
        Ensures the proposed shares value is valid.

        Arguments
        ---------

        shares: float, required
            The initial number of shares of a proposed asset.

        label : str, required
            The asset's type -- core, satellite, or benchmark. Benchmark assets
            can't have an initial share count other than 0.
        '''
        if shares < 0:
            raise ValueError('Only positive `shares` values are allowed.')
        elif label == 'benchmark' and shares != 0:
            raise ValueError("Only assets with a `label` of 'core' or "
                             "'satellite' can be initialized with a nonzero "
                             "number of shares.")

        return shares

    def _get_label_weights(self, label):
        '''
        Returns an array of weights for assets in the portfolio with a certain
        label. The sum of the array gives the total fraction currently taken up
        by those assets in their portfolio.

        Arguments
        ---------

        label : str, required
            The assets' type -- core, satellite, or benchmark.
        '''
        return np.array([val['fraction'] for val in self.assets.values()
                         if val['label'] == label])

    def _get_label_tickers(self, label):
        '''
        Return a list of all tickers with the given label.

        Arguments
        ---------

        label : str, required
            The assets' type -- core, satellite, or benchmark.
        '''
        return [tk for tk, val in self.assets.items() if val['label'] == label]

    def _get_check_printout(self, label):
        '''
        Used in self.check_assets().

        Print out weighting info for each ticker with a given label value.

        For satellite assets, returns the number currently present in the
        portfolio (0, 1, or 2). For other labels, returns their current total
        weight via self._get_label_weights().

        Arguments
        ---------

        label : str, required
            The assets' type -- core, satellite, or benchmark.
        '''
        print(f"{label} assets and target holding fraction(s):")
        ticks = self._get_label_tickers(label)

        if label != 'satellite':
            if len(ticks) > 0:
                fracs = self._get_label_weights(label)
                for i, fr in enumerate(fracs):
                    print(f"{fr*100:.5f}% in {ticks[i]}")

                label_count = fracs.sum()
                print(f"*** {label_count*100:.5f}% in {label} overall ***")
            else:
                label_count = 0
                print('None.')
        else:
            if len(ticks) > 0:
                for tk in ticks:
                    in_mkt = self.assets[tk]['in_mkt']
                    print(f"{'in' if in_mkt else 'out-of'}-market asset: "
                          f"{'    ' if in_mkt else ''}{tk}")
            else:
                print('None.')
            label_count = len(ticks)
            print(f"*** {self.sat_frac*100:.5f}% in {label} overall ***")

        print('----------------')
        return label_count

    def add_ticker(self, ticker, fraction=None, in_market=None, label=None,
                   shares=0, **kwargs):
        '''
        Create a new entry in self.assets for the given ticker. Adds a row of
        information about the result to self.tick_info. Please read "Arguments."

        Arguments
        ---------

        ticker : str, required
            The symbol of your desired asset.

        fraction : float, explained below.

        in_market : boolean, explained below.

        label : str, required
            The asset's type -- core, satellite, or benchmark. Choices for the
            other arguments change depending on your choice here.

            If `label` == 'core' or is left blank:
                -- A valid `fraction` value between 0 and 1 (inclusive) must be
                specified. (Note that if self.relative_core_frac == True, the
                `fraction` value for core tickers in self.assets will be the one
                entered by the user *multiplied by self.sat_frac.*)

            If `label` == 'satellite':
                -- `in_market` must be set True or False to indicate which
                asset to go to when a strategy sends a positive signal and
                which to retreat to when the strategy sends a negative signal.

            If `label` == 'benchmark':
                -- The same rules apply as for core assets, although `fraction`
                will always be entered as-is since `sat_frac` has no effect on
                benchmark assets.

        shares : float, optional
            The number of shares of this ticker that are held at the start of
            the eventual simulation. Only valid for assets with 'core' or
            'satellite' label. Useful for calculating rebalances for accounts
            that already exist. [default: 0]

        **kwargs : str, optional
            You may also add custom keys, perhaps to help with custom Strategy
            classes.
        '''
        ticker = self._validate_ticker(ticker)

        # create dict entry for this asset and save its label
        tick = {}
        label = self._validate_label(label, in_market)
        tick['label'] = label

        # add fraction or in_market to dict entry (depending on label)
        # (no "else: Error" condition needed because label was validated above)
        if label == 'satellite':
            tick['in_mkt'] = in_market
        else: # 'core' or 'benchmark'
            fraction = self._validate_fraction(fraction)
            tick['fraction'] = (fraction if label == 'benchmark'
                                or not self.relative_core_frac
                                else np.round(fraction * (1-self.sat_frac), 6))

        # for core or satellite assets, save initial share count to dict entry
        # go with Decimal here instead of float???
        shares = self._validate_shares(shares, label)
        tick['shares'] = shares

        # if any, add kwargs to dict entry
        for key, val in kwargs.items():
            tick.update({key: val})

        # add this information to the main assets dictionary
        self.assets[ticker] = tick

        # update the DataFrame with ticker start/end date info
        valid_tix = self._valid_tix
        nu_df = self.tick_info.append(valid_tix[valid_tix['ticker'] == ticker])
        self.tick_info = nu_df

    def edit_ticker_fraction(self, ticker, value):
        '''
        Change the fraction allocated to a core or benchmark asset that's
        already been added to self.assets. If self.relative_core_frac == True,
        core asset fractions will be adjusted relative to self.sat_frac.

        Arguments
        ---------

        ticker : str, required
            The symbol of the core or benchmark asset whose fraction you'd like
            to change.

        value : float, required
            The new fraction, which should be between 0 and 1 (inclusive).
        '''
        value = self._validate_fraction(value)

        # for core assets, adjust fraction relative to sat_frac if needed
        if self.relative_core_frac and self.assets[ticker]['label'] == 'core':
            value = np.round(value * (1 - self.sat_frac), 6)

        self.assets[ticker]['fraction'] = value

    def edit_ticker_mkt_status(self, ticker, value):
        '''
        Change the in-market status of a satellite asset that's already been
        added to self.assets.

        Arguments
        ---------

        ticker : str, required
            The symbol of your desired satellite asset.

        value : boolean, required
            The ticker's market status. Is it in-market or out-of-market?
        '''
        if self.assets[ticker]['label'] != 'satellite':
            raise ValueError('This method only modifies satellite assets.')

        value = self._validate_in_market(value)
        self.assets[ticker]['in_mkt'] = value

    def remove_ticker(self, ticker):
        '''
        Remove a ticker's info from self.assets and self.tick_info.

        Arguments
        ---------

        ticker : str, required
            The symbol of the asset you'd like to remove.
        '''
        self.assets.pop(ticker)
        self.tick_info = self.tick_info[self.tick_info != ticker]

    def reset_assets(self):
        '''
        Completely clear self.assets and self.tick_info of all information.
        '''
        self.assets = {}
        self.tick_info = pd.DataFrame([])

    def check_assets(self):
        '''
        Used in __init__() of HistoricalSimulator and can also be used
        independently.

        Checks fractions allocated to all tickers in self.assets, first for the
        main core/satellite portfolio and then for the benchmark portfolio. If
        all tests pass, self.assets is ready for use in a historical simulation.
        '''
        core_frac = self._get_check_printout('core')
        num_sat = self._get_check_printout('satellite')

        if np.round(core_frac + self.sat_frac, 5) != 1:
            raise ValueError('Make sure core and satellite fractions add up '
                             'to 1 (100%) before moving on to simulations.')

        if num_sat == 0 and self.sat_frac != 0:
            raise ValueError(f"{self.sat_frac*100:.5f}% of the portfolio is "
                             'allocated to the satellite, but no satellite '
                             'assets have been chosen. Set `sat_frac` to 0 or '
                             'choose two satellite assets.')
        elif num_sat == 1:
            raise ValueError('You may either have zero or two (in and out of '
                             'market) satellite assets.')
        elif num_sat == 2 :
            sat_ticks = self._get_label_tickers('satellite')
            mkt_mix = np.sum([self.assets[tk]['in_mkt'] for tk in sat_ticks])

            if mkt_mix != 1:
                raise ValueError('When including satellite assets, one of them '
                                 'must have in_mkt=False and the other must '
                                 'have in_mkt=True.')
            if self.sat_frac == 0:
                warnings.warn('Two satellite assets have been chosen, but '
                              f"`sat_frac` is 0%. Is that intentional?")

        print('-----passed-----')
        print('----------------')
        bench_frac = self._get_check_printout('benchmark')
        bench_ticks = self._get_label_tickers('benchmark')

        if not (   (bench_frac == 0 and len(bench_ticks) == 0)
                or (bench_frac == 1)  ):
            raise ValueError('If using benchmark assets, make sure their '
                             'fractions add up to 1 (100%) before running '
                             'simulations.')

        print('-----passed-----\n')
