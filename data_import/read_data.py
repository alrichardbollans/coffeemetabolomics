import os.path

import pandas as pd
from pkg_resources import resource_filename
from typing import Tuple

_input_path = resource_filename(__name__, "inputs")
_output_path = resource_filename(__name__, "outputs")

SPECIES = ['Arabica', 'Canephora', 'Stenophylla']


my_color_map = {'Arabica': "#1b9e77", 'Robusta': "#d95f02", 'Stenophylla': '#7570b3'}

manual_number_df = pd.read_csv(os.path.join(_input_path, 'AssignedCompoundsNumberedRT.csv'))
COMPS_OF_INTEREST = manual_number_df['Name'].values.tolist()
_IMPORTANT_COMPS_Unordered = ['Caffeine', 'Trigonelline', 'Sucrose', 'Citric acid', 'Theacrine', 'Mozambioside',
                   'CATR I', 'N-Eicosanoylserotonin', 'Dimethoxycinnamoylcaffeoylquinic acid II', '5-O Caffeoylquinic acid',
                   '4,5-di-O-caffeoylquinic acid',
                   '4-O-caffeoyl-3-O-ferroyloylquinic acid']
IMPORTANT_COMPS = [c for c in COMPS_OF_INTEREST if c in _IMPORTANT_COMPS_Unordered]
IMPORTANT_FLAVOUR_COMPS = ['Sucrose', 'Caffeine', 'Citric acid', 'Trigonelline']



numbering = dict(zip(manual_number_df['Name'], manual_number_df['Unnamed: 0']))

import csv

with open(os.path.join(_input_path, 'compound_enumeration.csv'), 'w') as f:
    w = csv.DictWriter(f, numbering.keys())
    w.writeheader()
    w.writerow(numbering)


def _parse_data(xl_file: str, tag: str, drop_raw_columns: bool = True):
    given_data = pd.read_excel(os.path.join(_input_path, xl_file))

    if drop_raw_columns:
        # Drop raw columns
        cols_to_drop = [c for c in given_data.columns if c.startswith('Area:')]
        given_data = given_data.drop(columns=cols_to_drop)

    given_data['temp_comp_id'] = ['comp' + str(c) for c in given_data.index]
    given_data['comp_id'] = given_data['Name'].fillna(given_data['temp_comp_id'])

    arabica_cols = [c for c in given_data.columns if '_arabica_' in c]
    canephora_cols = [c for c in given_data.columns if '_canephora_' in c]
    stenophylla_cols = [c for c in given_data.columns if '_stenophylla_' in c]

    print(arabica_cols)
    print(canephora_cols)
    print(stenophylla_cols)

    print(f'Number of arabica inputs: {len(arabica_cols)}')
    print(f'Number of canephora inputs: {len(canephora_cols)}')
    print(f'Number of stenophylla inputs: {len(stenophylla_cols)}')

    # assert len(arabica_cols)==33

    transposed_data = given_data.transpose()
    transposed_data.columns = transposed_data.loc['comp_id']
    transposed_data = transposed_data.drop('comp_id')

    arabica_data = transposed_data[transposed_data.index.str.contains('_arabica_')]
    arabica_data['species'] = 'Arabica'
    canephora_data = transposed_data[transposed_data.index.str.contains('_canephora_')]
    canephora_data['species'] = 'Robusta'

    stenophylla_data = transposed_data[transposed_data.index.str.contains('_stenophylla_')]
    stenophylla_data['species'] = 'Stenophylla'

    given_data.to_csv(os.path.join(_output_path, tag, 'parsed_data.csv'))
    transposed_data.to_csv(os.path.join(_output_path, tag, 'transposed_data.csv'))
    given_data.describe(include='all').to_csv(os.path.join(_output_path, tag, 'data_summary.csv'))
    transposed_data.describe(include='all').to_csv(os.path.join(_output_path, tag, 'transposed_data_summary.csv'))

    for c in arabica_data.columns:
        if c != 'species':
            arabica_data[c] = pd.to_numeric(arabica_data[c])
            canephora_data[c] = pd.to_numeric(canephora_data[c])
            stenophylla_data[c] = pd.to_numeric(stenophylla_data[c])

    arabica_data.to_csv(os.path.join(_output_path, tag, 'arabica_data.csv'))
    arabica_data.describe(include='all').to_csv(os.path.join(_output_path, tag, 'arabica_data_summary.csv'))
    canephora_data.to_csv(os.path.join(_output_path, tag, 'canephora_data.csv'))
    canephora_data.describe(include='all').to_csv(os.path.join(_output_path, tag, 'canephora_data_summary.csv'))

    stenophylla_data.to_csv(os.path.join(_output_path, tag, 'stenophylla_data.csv'))
    stenophylla_data.describe(include='all').to_csv(os.path.join(_output_path, tag, 'stenophylla_data_summary.csv'))


def read_norm_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    given_data = pd.read_csv(os.path.join(_output_path, 'normalised', 'parsed_data.csv'), index_col=0)
    transposed_data = pd.read_csv(os.path.join(_output_path, 'normalised', 'transposed_data.csv'), index_col=0)
    arabica_data = pd.read_csv(os.path.join(_output_path, 'normalised', 'arabica_data.csv'), index_col=0)
    canephora_data = pd.read_csv(os.path.join(_output_path, 'normalised', 'canephora_data.csv'), index_col=0)
    stenophylla_data = pd.read_csv(os.path.join(_output_path, 'normalised', 'stenophylla_data.csv'), index_col=0)

    return given_data, transposed_data, arabica_data, canephora_data, stenophylla_data


if __name__ == '__main__':
    _parse_data('CoffeeCompoundsQC37pc.xlsx', 'normalised')
    print(f'Number of compounds of interest: {len(COMPS_OF_INTEREST)}')
    print(f'Number of important comps: {len(IMPORTANT_COMPS)}')
    print(f'Number of important flavour comps: {len(IMPORTANT_FLAVOUR_COMPS)}')

    with open(os.path.join(_input_path, 'compound_names.txt'), 'w') as the_file:
        the_file.write(f'COMPS_OF_INTEREST:{COMPS_OF_INTEREST}\n')
        the_file.write(f'IMPORTANT_COMPS:{len(IMPORTANT_COMPS)}\n')
        the_file.write(f'IMPORTANT_FLAVOUR_COMPS:{IMPORTANT_FLAVOUR_COMPS}\n')
