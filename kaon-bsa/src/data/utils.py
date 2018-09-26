#
# Author: David Riser
# Date:   July 2, 2018
#
# File:   utils.py
# Description: This file contains utility functions
# that are used in bsa.py
#

import json
import logging
import numpy as np
import pandas as pd


def load_dataset(config):

    log = logging.getLogger(__name__)

    # Load data and drop nan values.
    data = pd.read_csv(config['file_path'],
                       compression=config['file_compression'],
                       nrows=config['sample_size'])
    data.dropna(how='any', inplace=True)

    log.info('Loaded dataset with size %d' % len(data))
    log.debug('Dataframe details: %s', data.info())

    # These axes will be kept, everything else will
    # be dropped.
    IMPORTANT_AXES = ['alpha', 'dist_cc', 'dist_cc_theta',
                      'dist_dcr1', 'dist_dcr3', 'dist_ecsf',
                      'dist_ec_edep', 'dist_ecu', 'dist_ecv',
                      'dist_ecw', 'dist_vz', 'helicity',
                      'missing_mass', 'p_mes', 'phi_h',
                      'pt', 'q2', 'x', 'z', 'dvz']

    # Perform the axis dropping.
    for col in data.columns:
        if col not in IMPORTANT_AXES:
            data.drop(col, axis=1, inplace=True)

    # Reduce memory usage and return loaded data for
    # analysis.
    data, _ = reduce_mem_usage(data)
    return data

def load_config(config_file):

    with open(config_file, 'r') as input_file:
        config = json.load(input_file)

        # Correct the type of our options.
        for opt in config.keys():
            if opt == 'sample_size':
                if config[opt] == 'None':
                    config[opt] = None
                else:
                    config[opt] = int(config[opt])
            if opt == 'n_bins':
                config[opt] = int(config[opt])
            if opt == 'z_range':
                config[opt] = [float(zp) for zp in config[opt]]

    return config

def build_filter(data, conf=None):
    '''
    data: This is the dataframe, we only need
    it to check that the variable is indeed there.

    conf: A dict that contains the
    cut name and the min, max values
    to be used.  Anything not in this dict
    will be assigned the nominal value.

    '''

    # When the filter is too long, pandas.DataFrame.query() breaks.
    # Use a vector of filters instead.
    filters = []

    # basic thing that always applies
    filters.append('q2 > 1.0')

    if 'w' in data.columns:
        filters.append('w > 2.0')

    if 'meson_id' in data.columns:
        filters.append('meson_id == 321')

    # nominal values
    nominal_conf = {}
    nominal_conf['alpha'] = [0.05, 1.0]
    nominal_conf['dist_cc'] = [-1.0, 1.0]
    nominal_conf['dist_cc_theta'] = [-1.0, 1.0]
    nominal_conf['dist_dcr1'] = [-1.0, 1.0]
    nominal_conf['dist_dcr3'] = [-1.0, 1.0]
    nominal_conf['dist_ecsf'] = [-1.0, 1.0]
    nominal_conf['dist_ecu'] = [-1.0, 1.0]
    nominal_conf['dist_ecv'] = [-1.0, 1.0]
    nominal_conf['dist_ecw'] = [-1.0, 1.0]
    nominal_conf['dist_ec_edep'] = [-1.0, 1.0]
    nominal_conf['dist_vz'] = [-1.0, 1.0]
    nominal_conf['missing_mass'] = [1.25, 5.0]
    nominal_conf['p_mes'] = [0.35, 5.0]
    nominal_conf['dvz'] = [-2.5, 2.5]

    # start adding the special options
    if conf:
        for k, v in conf.iteritems():

            # these have to be valid
            if len(v) is not 2:
                print('Improper limits for parameter %s' % v)
                return filters

            if k in data.columns:
                filters.append('%s > %f and %s < %f' % (k, v[0], k, v[1]))
            # print('OPTION: %s, LIMITS: [%f,%f]' % (k,v[0],v[1]))
            else:
                print('Problem adding filter for %s because it is not in the dataframe.columns' % k)

        # now add the default options for those which were not specified
        for k, v in nominal_conf.iteritems():
            if k not in conf.keys():
                if k in data.columns:
                    filters.append('%s > %f and %s < %f ' % (k, v[0], k, v[1]))
                else:
                    print('Problem adding filter for %s because it is not in the dataframe.columns' % k)
            else:
                pass
                #                print('Not adding nominal cut for %s, it was in the special cuts.' % k)


    else:
        for k, v in nominal_conf.iteritems():
            if k in data.columns:
                #                print('OPTION: %s, LIMITS: [%f,%f]' % (k,v[0],v[1]))
                filters.append('%s > %f and %s < %f ' % (k, v[0], k, v[1]))
            else:
                print('Problem adding filter for %s because it is not in the dataframe.columns' % k)

    return filters


def build_dataframe(data, filters):
    CHUNK_SIZE = 4

    if len(filters) < CHUNK_SIZE:
        return data.query(' and '.join(filters))

    else:
        d = data.copy(deep=True)

        for i in range(0, len(filters), CHUNK_SIZE):
            f = filters[i:i + CHUNK_SIZE]
            d.query(' and '.join(f), inplace=True)

        return d


def reduce_mem_usage(props):
    ''' Taken from Kaggle, if I can find the kernel
    I will give proper credit.  If function is yours
    please let me know I will give you credit.

    Reduces dataframe memory consumption by
    changing dtype to lowest memory usage variant
    that will fit the data column.

    '''
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

    return props, NAlist

def bin_by_quantile(data, axis=None, n_bins=None):
    # find minimum and maximum
    axis_range = np.min(data[axis]), np.max(data[axis])

    # step in quantile to do binning
    quantile_step = 1.0 / n_bins

    bins = []
    for index in range(n_bins + 1):
        bins.append(data[axis].quantile(index * quantile_step))

    return np.array(bins)

def convert_bin_limits_to_centers(limits):
    centers = []
    for i in range(len(limits)-1):
            centers.append(limits[i] + 0.5*(limits[i+1]-limits[i]))

    return np.array(centers)


def get_asymmetry_df(data, axis, n_bins,
                     beam_pol=0.749,
                     beam_pol_unc=0.024,
                     n_phi_bins=12,
                     custom_bin_limits=None):
    # setup the binning for the phi axis
    phi_bin_limits = np.linspace(-180, 180, n_phi_bins + 1)

    # covert bin limits to central positions for plotting
    phi_bin_centers = convert_bin_limits_to_centers(phi_bin_limits)

    if custom_bin_limits is not None:
        axis_range = list([custom_bin_limits[0], custom_bin_limits[-1]])
        bin_limits = custom_bin_limits
        n_bins = len(custom_bin_limits) - 1

    else:
        # calculate range of binned axis
        axis_range = data[axis].quantile(0.001), data[axis].quantile(0.999)

        # create bin limits for binning up the dataframe
        bin_limits = np.linspace(axis_range[0], axis_range[1], n_bins + 1)

    results = []
    for index in range(len(bin_limits) - 1):
        # setup a string used to query for data in this bin
        # and the corresponding title for this bin
        bin_query = ('%s > %f and %s < %f' % (axis, bin_limits[index], axis, bin_limits[index + 1]))
        bin_title = ('%s $\in [%.2f, %.2f]$' % (axis, bin_limits[index], bin_limits[index + 1]))

        # query the data for this bin
        data_subset = data.query(bin_query)

        # get histograms for positive and negative helicity
        pos_counts, _ = np.histogram(data_subset[data_subset.helicity > 0].phi_h, bins=phi_bin_limits)
        neg_counts, _ = np.histogram(data_subset[data_subset.helicity < 0].phi_h, bins=phi_bin_limits)

        # calculate the asymmetry and the error
        diff = np.array(pos_counts - neg_counts, dtype=np.float32)
        total = np.array(pos_counts + neg_counts, dtype=np.float32)
        asymmetry = diff / total / beam_pol
        error = np.sqrt((1 - asymmetry ** 2) / total)
        sys0 = beam_pol_unc * np.abs(asymmetry)

        result = {}
        result['axis'] = [axis] * len(pos_counts)
        result['axis_min'] = [bin_limits[index]] * len(pos_counts)
        result['axis_max'] = [bin_limits[index + 1]] * len(pos_counts)
        result['axis_bin'] = [index] * len(pos_counts)
        result['counts_pos'] = pos_counts
        result['counts_neg'] = neg_counts
        result['value'] = asymmetry
        result['stat'] = error
        result['sys_0'] = sys0
        result['phi'] = phi_bin_centers
        result['phi_bin'] = np.arange(len(phi_bin_centers))
        results.append(pd.DataFrame(result))

    return pd.concat(results)


def get_results(data, bins, config):
    df_store = []

    for axis in config['axes']:
        if axis != 'z':
            df_store.append(get_asymmetry_df(data=data.query('z > %f and z < %f' % (config['z_range'][0], config['z_range'][1])),
                                             axis=axis,
                                             n_bins=len(bins[axis]),
                                             custom_bin_limits=bins[axis],
                                             n_phi_bins=12)
                            )
        else:
            df_store.append(get_asymmetry_df(data=data,
                                             axis=axis,
                                             n_bins=len(bins[axis]),
                                             custom_bin_limits=bins[axis],
                                             n_phi_bins=12)
                            )

    for df in df_store:
        df['global_index'] = df.phi_bin + df.axis_bin * len(np.unique(df.phi_bin))

    return pd.concat(df_store)


def save_to_database(results,
                     db_path='database/phi/',
                     naming_scheme='variation_%s_%s.csv'):
    for parameter in results.keys():
        if parameter is not 'nominal':
            for level in results[parameter].keys():
                output_path = db_path + naming_scheme % (parameter, level)
                results[parameter][level].to_csv(output_path, index=False)


def get_random_config(data, variations):
    random_configuration = {}
    for parameter_name, stricts in variations.iteritems():
        min_strict = min(stricts.keys())
        max_strict = max(stricts.keys())

        minimum = np.random.uniform(stricts[min_strict][0],
                                    stricts[max_strict][0])
        maximum = np.random.uniform(stricts[min_strict][1],
                                    stricts[max_strict][1])

        random_configuration[parameter_name] = [minimum, maximum]

    return build_filter(data, random_configuration)


def read_random_results(path_to_db='database/random/phi'):
    random_results = {}

    database_files = glob.glob(path_to_db + '/*.csv')

    for database_file in database_files:
        random_results[database_file] = pd.read_csv(database_file)

    return random_results


def get_global_bin_data(nominal, variations, axis, global_index):
    d = nominal.query('axis == "%s" and global_index == %d' % (axis, global_index))

    v = {}
    for var in variations.keys():
        v[var] = variations[var].query('axis == "%s" and global_index == %d' % (axis, global_index))

    found_data = True

    if len(d) is not 1:
        found_data = False

    for vi in v.keys():
        if len(v[vi]) is not 1:
            found_data = False

    if not found_data:
        print('Trouble finding data for global index %d' % global_index)
        return

        # success
    return d, v


def get_largest_shifts(results):
    '''
    inputs
    ------
    results - A dictionary generated above which contains results of
    nominal running, as well as parameter variations.

    outputs
    -------

    '''

    # we need to do the process for each axis independently
    active_axes = np.unique(results['nominal'].axis)
    n_global = len(np.unique(results['nominal'].global_index))

    # something to store the result in
    df_dict = {}
    df_dict['axis'] = []
    df_dict['global_index'] = []

    # somewhere to correlate the variation name with the column name
    column_dict = {}

    # going though the different variations
    i_par = 1
    for par in results.keys():
        if par is not 'nominal':
            # setup somewhere to store this
            column_title = 'sys_%d' % i_par
            df_dict[column_title] = []
            column_dict[column_title] = par
            i_par += 1

    for axis in active_axes:
        for index in range(n_global):
            df_dict['axis'].append(axis)
            df_dict['global_index'].append(index)

            i_par = 1
            for par in results.keys():
                if par is not 'nominal':
                    # setup somewhere to store this
                    column_title = 'sys_%d' % i_par

                    d, v = get_global_bin_data(results['nominal'], results[par], axis, index)
                    current_shifts = [val['shift'].values[0] for key, val in v.iteritems()]
                    df_dict[column_title].append(np.max(np.abs(current_shifts)))
                    i_par += 1

    # now add them in quadrature
    df_dict['sys_total'] = []
    for i in range(len(df_dict['global_index'])):

        bin_total = 0.0
        for k in df_dict.keys():
            if 'sys' in k and 'total' not in k:
                bin_total += df_dict[k][i] ** 2

        df_dict['sys_total'].append(bin_total)

    df_dict['sys_total'] = np.sqrt(df_dict['sys_total'])
    df = pd.DataFrame(df_dict)
    return df, column_dict


def get_linearized_error(results):
    '''
    inputs
    ------
    results - A dictionary generated above which contains results of
    nominal running, as well as parameter variations.

    outputs
    -------

    '''

    # we need to do the process for each axis independently
    active_axes = np.unique(results['nominal'].axis)
    n_global = len(np.unique(results['nominal'].global_index))

    # something to store the result in
    df_dict = {}
    df_dict['axis'] = []
    df_dict['global_index'] = []

    # somewhere to correlate the variation name with the column name
    column_dict = {}

    # going though the different variations
    i_par = 1
    for par in results.keys():
        if par is not 'nominal':
            # setup somewhere to store this
            column_title = 'sys_%d' % i_par
            df_dict[column_title] = []
            column_dict[column_title] = par
            i_par += 1

    for axis in active_axes:
        for index in range(n_global):
            df_dict['axis'].append(axis)
            df_dict['global_index'].append(index)

            i_par = 1
            for par in results.keys():
                if par is not 'nominal':
                    # setup somewhere to store this
                    column_title = 'sys_%d' % i_par

                    d, v = get_global_bin_data(results['nominal'], results[par], axis, index)

                    if 1 in v.keys() and -1 in v.keys():
                        delta = (v[1]['value'].values[0] - v[-1]['value'].values[0])
                    elif 0 in v.keys() and -1 in v.keys():
                        delta = (v[0]['value'].values[0] - v[-1]['value'].values[0])
                    elif 0 in v.keys() and 1 in v.keys():
                        delta = (v[1]['value'].values[0] - v[0]['value'].values[0])
                    else:
                        raise ValueError('For parameter %s I dont know how to linearize, there are no shifts?' % par)

                    df_dict[column_title].append(delta)
                    i_par += 1

    # now add them in quadrature
    df_dict['sys_total'] = []
    for i in range(len(df_dict['global_index'])):

        bin_total = 0.0
        for k in df_dict.keys():
            if 'sys' in k and 'total' not in k:
                bin_total += df_dict[k][i] ** 2

        df_dict['sys_total'].append(bin_total)

    df_dict['sys_total'] = np.sqrt(df_dict['sys_total'])
    df = pd.DataFrame(df_dict)
    return df, column_dict


def get_global_bin(results, axis, bin):
    r = []

    for key, value in results.iteritems():
        r.append(value.query('axis == "%s" and global_index == %d' % (axis, bin)).value.values[0])

    return np.array(r, dtype=np.float32)


def get_randomized_error(results):
    '''
    inputs
    ------
    results - A dictionary generated above which contains results of
    random running.

    outputs
    -------
    df - A dataframe indexed by axis and global_bin number that contains
    the expectation value and standard deviation of the random measurements.

    '''

    # we need to do the process for each axis independently
    test_configuration_name = results.keys()[-1]
    active_axes = np.unique(results[test_configuration_name].axis)
    n_global = len(np.unique(results[test_configuration_name].global_index))

    # something to store the result in
    df_dict = {}
    df_dict['axis'] = []
    df_dict['global_index'] = []
    df_dict['exp_val'] = []
    df_dict['std_dev'] = []

    for axis in active_axes:
        for bin in range(n_global):
            samples = get_global_bin(results, axis, bin)

            df_dict['axis'].append(axis)
            df_dict['global_index'].append(bin)
            df_dict['exp_val'].append(np.average(samples))
            df_dict['std_dev'].append(np.std(samples))

    df = pd.DataFrame(df_dict)
    return df



