import networkx as nx
import os
import pickle
import json

import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
from glob import glob
from csv import reader


def set_dict():
    return defaultdict(set)


def dict_dict():
    return defaultdict(dict)


def generate_network_gexf(database_name=None,
                db_config_file=None,
                input_json_dir=None,
                output_network_file=None,
                save_pkl=True,
                load_from_pkl=True,
                load_from_gexf=False,
                input_network_file=None,
                dict_pkl_file=None,
                users_pkl_file=None,
                mutual_pkl_file=None,
                table_name='tweets',
                connection_type='retweet',
                link_type='mutual',
                edge_weight=True,
                conditions=[],
                attributes=None,
                label='screen_name',
                connection_limit=10,
                mutual_limit=5,
                network_pruning=10,
                itersize=1000,
                limit=None,
                output_type='gexf',
                overwrite=False,
                mutual_overwrite=True,
                verbose=False):

    # Type of tweet requested. Only compatible with Twitter data.
    if connection_type not in ['retweet', 'quote', 'reply', 'mention', 'all']:
        raise ValueError(f'connection_type must be retweet, quote, reply, mention, all, -- not, {connection_type}')

    # This is for output types.
    if output_type not in ['gexf', 'dynamic']:
        raise ValueError(f'output_type must be networkx, dynamic, not, {output_type}')

    # If a graph exists, use it.
    graph = None
    if not overwrite and load_from_gexf and os.path.exists(input_network_file):
        print('Graph file already exists and overwrite=False. Skipping.')

    # If cached files exist, use them.
    elif not overwrite and load_from_pkl and os.path.exists(dict_pkl_file) and os.path.exists(users_pkl_file):

        if verbose:
            print('Loading input dict')

        with open(dict_pkl_file, 'rb') as openfile:
            connections_dict = pickle.load(openfile)

        if verbose:
            print('Loading user dict')

        with open(users_pkl_file, 'rb') as openfile:
            user_dict = pickle.load(openfile)

        if mutual_pkl_file is not None and link_type == 'mutual':
            if os.path.exists(mutual_pkl_file):
                if verbose:
                    print('Loading mutual connections dict')

                with open(mutual_pkl_file, 'rb') as openfile:
                    mutual_dict = pickle.load(openfile)
            else:
                mutual_dict = None
        else:
            mutual_dict = None

    else:

        raise ValueError("Something's gone wrong, and intermediate data files (dict_pkl_file, users_pkl_file) were not provided to this function.")

        if input_json_dir:
            connections_dict, user_dict, = load_connection_data(input_json_dir,
                    output_network_file, save_pkl, dict_pkl_file, users_pkl_file, 
                    connection_type, attributes, label)
        else:
            connections_dict, user_dict, = stream_connection_data(database_name,
                    db_config_file, output_network_file,
                    save_pkl, dict_pkl_file, users_pkl_file,
                    table_name, connection_type, conditions,
                    attributes, label, itersize,
                    limit)

    if output_type == 'gexf':
        if graph is None:
            graph = process_dicts_nx(connections_dict, user_dict, connection_limit, link_type, mutual_dict,
                mutual_pkl_file, mutual_overwrite, edge_weight, mutual_limit)
            nx.write_gexf(graph, output_network_file)

    return graph


def load_csv_connection_data(input_csv, output_network_file,
                        save_pkl=True, dict_pkl_file=None, users_pkl_file=None,
                        attributes=None, verbose=True):
    if verbose:
        print("Entering load method")
    connections_dict = defaultdict(dict_dict)
    username_dict = dict()

    if verbose:
        print("Reading CSV..")

    with open(input_csv, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)

        if header is not None:
            header = [x.lower() for x in header]
            if 'source' in header:
                source_index = header.index('source')
            else:
                print('ERROR: NO SOURCE COLUMN')
                source_index = 0
            if 'target' in header:
                target_index = header.index('target')
            else:
                print('ERROR: NO TARGET COLUMN')
                target_index = 1
            if 'source_label' in header:
                source_label_index = header.index('source_label')
            else:
                source_label_index = source_index
            if 'target_label' in header:
                target_label_index = header.index('target_label')
            else:
                target_label_index = target_index
            for row in tqdm(csv_reader):
                screen_name = row[source_label_index]
                connect_user_name = row[target_label_index]
                user_id = row[source_index]
                connect_user = row[target_index]
                username_dict[user_id] = screen_name
                username_dict[connect_user] = connect_user_name
                if connect_user not in connections_dict[user_id]:
                    connections_dict[user_id][connect_user] = 0
                connections_dict[user_id][connect_user] += 1

    if verbose:
        print("Finished reading.")
    

    if verbose:
        print(len(connections_dict), 'connecting users')
        print(len(username_dict), 'users total')

    if save_pkl:
        with open(dict_pkl_file, 'wb') as openfile:
            pickle.dump(connections_dict, openfile)
        with open(users_pkl_file, 'wb') as openfile:
            pickle.dump(username_dict, openfile)

    return connections_dict, username_dict


def load_connection_data(input_json_dir,
                    output_network_file,
                    save_pkl=True, dict_pkl_file=None, users_pkl_file=None, connection_type='retweet',
                    attributes=None, label='screen_name'):

    json_files = glob(os.path.join(input_json_dir, '*.json*'))
    connections_dict = defaultdict(dict_dict)
    username_dict = defaultdict(set_dict)

    if connection_type == 'all':
        connection_type = ['retweet', 'quote', 'reply']
    if isinstance(connection_type, str):
        connection_type = [connection_type]
    if isinstance(attributes, str):
        attributes = [attributes]

    for json_file in tqdm(json_files):

        items = load_json(json_file)

        for item in items:
            # pprint(item)
            user_id = item['user']['id']
            screen_name = item['user']['screen_name']

            connect_users = []
            connect_screen_names = []
            for connect in connection_type:
                connect_user = None
                if connect == 'retweet' and 'retweeted_status' in item:
                    connect_users += [item['retweeted_status']['user']['id']]
                    connect_screen_names += [item['retweeted_status']['user']['screen_name']]
                if connect == 'quote' and 'quoted_status' in item:
                    connect_users += [item['quoted_status']['user']['id']]
                    connect_screen_names += [item['quoted_status']['user']['screen_name']]
                if connect == 'reply' and item['in_reply_to_screen_name'] is not None:
                    connect_users += [item['in_reply_to_user_id']]
                    connect_screen_names += [item['in_reply_to_screen_name']]
                if connect == 'mention':
                    if 'user_mentions' in item['entities']:
                        user_dict = {x['id']: x['screen_name'] for x in item['entities']['user_mentions']}
                        text = item['full_text']
                        if 'retweeted_status' in item:
                            text = str.split(text, 'RT ')[0]  # This will fail sometimes

                        if user_dict:
                            # Redundant
                            user_ids = [x for x in user_dict if f'@{user_dict[x]}' in text]
                            for uid in user_ids:
                                connect_users += [uid]
                                connect_screen_names += [user_dict[uid]]

            if connect_users:
                username_dict[user_id]['screen_name'].add(screen_name)
                for connect_user, connect_screen_name in zip(connect_users, connect_screen_names):
                    if 'count' not in connections_dict[user_id][connect_user]:
                        connections_dict[user_id][connect_user]['count'] = 0
                    connections_dict[user_id][connect_user]['count'] += 1
                    username_dict[connect_user]['screen_name'].add(connect_screen_name)
                    if attributes is not None:
                        for attribute in attributes:
                            connections_dict[user_id][connect_user][attribute] = item[attribute]

    if save_pkl:
        with open(dict_pkl_file, 'wb') as openfile:
            pickle.dump(connections_dict, openfile)
        with open(users_pkl_file, 'wb') as openfile:
            pickle.dump(username_dict, openfile)

    return connections_dict, username_dict


def process_dicts_nx(input_dict, user_dict, connection_limit=20,
            connection_mode='mutual', mutual_dict=None,
            mutual_pkl_file=None, mutual_overwrite=False,
            edge_weight=True, mutual_limit=5):

    if connection_mode == 'direct':

        graph = nx.DiGraph()

        for connecting_user, connecting_dict in tqdm(input_dict.items()):
            for connected_user, connected_dict in connecting_dict.items():
                if connecting_user == connected_user:
                    continue
                if connected_dict['count'] >= connection_limit:
                    
                    if edge_weight:
                        graph.add_edge(connecting_user, connected_user, weight=connected_dict['count'])
                    else:
                        graph.add_edge(connecting_user, connected_user)

                    for key, value in connected_dict.items():
                        if key != 'count':
                            graph[connecting_user][connected_user][key] = value
                    graph.add_node(connecting_user, label=next(iter(user_dict[connecting_user]['screen_name'])), dataset=user_dict[connecting_user]['dataset'])
                    graph.add_node(connected_user, label=next(iter(user_dict[connected_user]['screen_name'])), dataset=user_dict[connected_user]['dataset']) 

    elif connection_mode == 'reciprocal':

        graph = nx.Graph()

        for connecting_user, connecting_dict in tqdm(input_dict.items()):
            for connected_user, connected_dict in connecting_dict.items():
                if connecting_user == connected_user:
                    continue
                connect_count = connected_dict['count']
                if connect_count >= connection_limit:
                    if connected_user not in input_dict:
                        continue
                    if connecting_user not in input_dict[connected_user]:
                        continue
                    connected_count = input_dict[connected_user][connecting_user]['count']
                    if connected_count >= connection_limit:

                        if edge_weight:
                            graph.add_edge(connecting_user, connected_user, weight=min(connect_count, connected_count))
                        else:
                            graph.add_edge(connecting_user, connected_user)

                        for key, value in connected_dict.items():
                            if key != 'count':
                                graph[connecting_user][connected_user][key] = value

                        graph.add_node(connecting_user, label=next(iter(user_dict[connecting_user]['screen_name'])))
                        graph.add_node(connected_user, label=next(iter(user_dict[connected_user]['screen_name']))) 

    elif connection_mode == 'mutual':

        graph = nx.Graph()

        if mutual_dict is None or mutual_overwrite:
            mutual_dict = defaultdict(int)
            pbar = tqdm(input_dict.items())
            for connecting_user, connecting_dict in pbar:
                connected_users = [key for key, val in connecting_dict.items() if val >= mutual_limit]
                pairs = combinations(connected_users, 2)
                pbar.set_description("Mutual dict %s" % len(mutual_dict))
                for pair in pairs:
                    mutual_dict[frozenset(pair)] += 1

            with open(mutual_pkl_file, 'wb') as openfile:
                pickle.dump(mutual_dict, openfile)

        for user_set, count in tqdm(mutual_dict.items()):
            [connecting_user, connected_user] = list(user_set)
            if count >= connection_limit:
                if edge_weight:
                    graph.add_edge(connecting_user, connected_user, weight=count)
                else:
                    graph.add_edge(connecting_user, connected_user)

                graph.add_node(connecting_user, label=user_dict[connecting_user])
                graph.add_node(connected_user, label=user_dict[connected_user])

    return graph


def load_json(input_json, output_type='dict'):

    # Should probably check this at the format level.
    if input_json.endswith('json'):
        with open(input_json, 'r') as f:
            data = json.load(f)
    else:
        with open(input_json, 'r') as f:
            data = [json.loads(jline) for jline in list(f)]

    return data