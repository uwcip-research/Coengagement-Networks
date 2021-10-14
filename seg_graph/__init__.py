import os
import subprocess
import sys
import glob
from .core import generate_network_gexf, load_csv_connection_data, load_connection_data


# This file manages the initial graph creation if a GEXF file is not supplied, then passes off the data to the Java
# code, where the actual graph layoutting and other operations are completed.
def main(**kwargs):
    # defaults that are used for initial GEXF generation.
    connection_type = 'retweet'
    graph_type = 'mutual'
    connection_limit = kwargs['connection_limit']  # ASK ANDREW
    mutual_limit = kwargs['mutual_limit']

    # This branch of the if handles when a GEXF file is already passed (which may or may not be shared-audience)
    # In this branch, we pretty much just hand off to Java immediately.
    if kwargs['output_all'] is not None:
        kwargs['output_graph'] = str(kwargs['output_all']) + '.gexf'
        kwargs['output_pdf'] = str(kwargs['output_all']) + '.pdf'
        kwargs['output_csv'] = str(kwargs['output_all']) + '.csv'

    if kwargs['input_graph'] is not None:
        # Initial flags to java
        cmd = ['java', '-Xmx6g', '-cp', '.:/home/uwcip/src/:/home/uwcip/src/gephi-toolkit-0.9.2-all.jar', 'GephiInterface', '--input', kwargs['input_graph']]

        # Setting up output(s)
        if kwargs['output_graph'] is not None:
            cmd.extend(['--output-graph', kwargs['output_graph']])
        if kwargs['output_pdf'] is not None:
            cmd.extend(['--output-pdf', kwargs['output_pdf']])
        if kwargs['output_csv']:
            cmd.extend(['--output-csv', kwargs['output_csv']])

        # Adding a reference file if needed, and optionally filtering only to those
        if kwargs['ref'] is not None:
            cmd.extend(['--reference', kwargs['ref']])
            if kwargs['trim_unconnected']:
                cmd.append('--filter')

        # Add a error file to route outputs from the code to
        if kwargs['err'] is not None:
            cmd.extend(['--error', kwargs['err']])

        # Add sizing parameters
        if kwargs['min-size'] is not None:
            cmd.extend(['--size-min', kwargs['min-size']])
        if kwargs['max-size'] is not None:
            cmd.extend(['--size-max', kwargs['max-size']])

        #Call Java code
        subprocess.run(cmd)

    else:  # input from data - requires pre-processing to create intermediate graph file

        data_path = kwargs['data_dir'] # Directory where temporary data is stored
        delete_temp_data = False
        if data_path is None: # Do not cache data, delete after running
            delete_temp_data = True
            data_path = os.path.join(os.getcwd(), 'temp_data') # Directory where data will be stored on running

        network_file = os.path.join(data_path, "Reference_Network.gexf") # Where we will store our initial GEXF file
        user_pkl = os.path.join(data_path, "user_connects.pkl") # Temporary files
        dict_pkl = os.path.join(data_path, "dict_connects.pkl") # Temporary files
        mutual_pkl = os.path.join(data_path, f"{mutual_limit}_mutual_connects.pkl") # Temporary files

        extension = kwargs['input_data'].split('.')[1] # Determine whether we're working with a CSV or a directory of jsons
        if extension == 'csv':
            load_csv_connection_data(
                kwargs['input_data'],
                network_file,
                True,
                dict_pkl,
                user_pkl
            )

        elif extension == 'json':
            load_connection_data(kwargs['input_data'],
                                 network_file, True, dict_pkl, user_pkl, connection_type='retweet')

        else:
            print('ERROR: Unsupported input type ' + extension)
            sys.exit(1)

        if kwargs['python_only']:
            generate_network_gexf(database_name=None,  # Generate temporary GEXF from temp files using -n and -s parameters
                                  db_config_file=None,
                                  input_json_dir=None,
                                  output_network_file=kwargs['output_graph'],
                                  save_pkl=True,
                                  load_from_pkl=True,
                                  load_from_gexf=False,
                                  input_network_file=None,
                                  dict_pkl_file=dict_pkl,
                                  users_pkl_file=user_pkl,
                                  mutual_pkl_file=mutual_pkl,
                                  table_name='',
                                  connection_type=connection_type,
                                  link_type=graph_type,
                                  conditions=[],
                                  attributes=None,
                                  label='screen_name',
                                  connection_limit=connection_limit,
                                  mutual_limit=mutual_limit,
                                  network_pruning=0,
                                  itersize=1000,
                                  limit=None,
                                  output_type='gexf',
                                  overwrite=False,
                                  mutual_overwrite=True)
        else:
            out_network = f'{network_file}_{graph_type}_{connection_limit}_{mutual_limit}_{connection_type}.gexf'  # Temp GEXF

            generate_network_gexf(database_name=None,  # Generate temporary GEXF from temp files using -n and -s parameters
                              db_config_file=None,
                              input_json_dir=None,
                              output_network_file=out_network,
                              save_pkl=True,
                              load_from_pkl=True,
                              load_from_gexf=False,
                              input_network_file=None,
                              dict_pkl_file=dict_pkl,
                              users_pkl_file=user_pkl,
                              mutual_pkl_file=mutual_pkl,
                              table_name='',
                              connection_type=connection_type,
                              link_type=graph_type,
                              conditions=[],
                              attributes=None,
                              label='screen_name',
                              connection_limit=connection_limit,
                              mutual_limit=mutual_limit,
                              network_pruning=0,
                              itersize=1000,
                              limit=None,
                              output_type='gexf',
                              overwrite=False,
                              mutual_overwrite=True)

            # Beginning of java command
            cmd = ['java', '-Xmx6g', '-cp', './:home/uwcip/src/:/home/uwcip/src/gephi-toolkit-0.9.2-all.jar', 'GephiInterface', '--input', out_network]

            # Output file(s)
            if kwargs['output_graph'] is not None:
                cmd.extend(['--output-graph', kwargs['output_graph']])
            if kwargs['output_pdf'] is not None:
                cmd.extend(['--output-pdf', kwargs['output_pdf']])
            if kwargs['output_csv']:
                cmd.extend(['--output-csv', kwargs['output_csv']])

            # Reference file
            if kwargs['ref'] is not None:
                cmd.extend(['--reference', kwargs['ref']])
                if kwargs['trim_unconnected']:
                    cmd.append('--filter')

            # Error stream
            if kwargs['err'] is not None:
                cmd.extend(['--error', kwargs['err']])

            # Add sizing parameters
            if kwargs['min_size'] is not None:
                cmd.extend(['--size-min', kwargs['min_size']])
            if kwargs['max_size'] is not None:
                cmd.extend(['--size-max', kwargs['max_size']])
            # Run Java code
            subprocess.run(cmd)

        # Cleanup temp files
        if (delete_temp_data):
            files = glob.glob(data_path+'/*')
            for file in files:
                try:
                    os.remove(file)
                except OSError as e:
                    print("Error: %s : %s" % (file, e.strerror))
