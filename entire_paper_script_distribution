import csv
import os
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from glob import glob
from itertools import combinations
from pprint import pprint

import shutil
import community as community_louvain
import gdown
# import matplotlib.dates as mdates
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from scipy.sparse import save_npz, load_npz
from tqdm import tqdm
from zipfile import ZipFile, ZIP_LZMA
import lzma
import tarfile


def int_dict():
    return defaultdict(int)


def dict_dict():
    return defaultdict(dict)


def generate_network_gexf(
    database_name=None,
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
    conditions=None,
    attributes=None,
    label='screen_name',
    connection_limit=10,
    mutual_limit=5,
    network_pruning=10,
    itersize=1000,
    limit=None,
    mode='networkx',
    overwrite=False,
    mutual_overwrite=True,
    skip_mutual=False,
):

    graph = None

    if not overwrite and load_from_gexf and os.path.exists(input_network_file):
        graph = nx.read_gexf(input_network_file)
    elif not overwrite and load_from_pkl:
        # Check for connections dictionary.
        if type(dict_pkl_file) is not str:
            connections_dict = dict_pkl_file
        elif os.path.exists(dict_pkl_file):
            print('Loading input dict')
            with open(dict_pkl_file, 'rb') as openfile:
                connections_dict = pickle.load(openfile)
        else:
            raise ValueError('Cannot load connections file ', dict_pkl_file)

        # Check for connections dictionary.
        if type(users_pkl_file) is not str:
            user_dict = users_pkl_file
        elif os.path.exists(users_pkl_file):
            print('Loading user dict')
            with open(users_pkl_file, 'rb') as openfile:
                user_dict = pickle.load(openfile)
        else:
            raise ValueError('Cannot load connections file ', users_pkl_file)

        # Check for mutual dictionary
        if mutual_pkl_file is not None and link_type == 'mutual':
            if type(mutual_pkl_file) is not str:
                mutual_dict = mutual_pkl_file
            elif os.path.exists(mutual_pkl_file):
                print('Loading mutual connections dict')
                with open(mutual_pkl_file, 'rb') as openfile:
                    mutual_dict = pickle.load(openfile)
            else:
                mutual_dict = None
        else:
            mutual_dict = None
    else:
        raise ValueError('No PKL files provided. Streaming from database not available in this code version.')

    if graph is None:
        graph = process_dicts_nx(
            connections_dict, user_dict, connection_limit, link_type, mutual_dict,
            mutual_pkl_file, mutual_overwrite, edge_weight, mutual_limit, skip_mutual,
        )
        nx.write_gexf(graph, output_network_file)

    return


def process_dicts_nx(
    input_dict,
    user_dict,
    connection_limit=20,
    connection_mode='mutual',
    mutual_dict=None,
    mutual_pkl_file=None,
    mutual_overwrite=False,
    edge_weight=True,
    mutual_limit=5,
    skip_mutual=False,
):

    if connection_mode == 'direct':

        graph = nx.DiGraph()

        for connecting_user, connecting_dict in tqdm(input_dict.items()):
            for connected_user, connected_dict in connecting_dict.items():
                if connecting_user == connected_user:
                    continue
                if connected_dict >= connection_limit:

                    if edge_weight:
                        graph.add_edge(connecting_user, connected_user, weight=connected_dict)
                    else:
                        graph.add_edge(connecting_user, connected_user)

                    graph.add_node(connecting_user, label=next(iter(user_dict[connecting_user])))
                    graph.add_node(connected_user, label=next(iter(user_dict[connected_user])))

    elif connection_mode == 'mutual':

        graph = nx.Graph()

        if skip_mutual:

            user_totals = defaultdict(int)
            pbar = tqdm(input_dict.items())
            for connecting_user, connecting_dict in pbar:
                connected_users = [key for key, val in connecting_dict.items() if val >= mutual_limit]
                for connected_user in connected_users:
                    user_totals[connected_user] += 1
            high_tweet_users = [key for key, val in user_totals.items() if val >= connection_limit]
            del user_totals

            mutual_dict = defaultdict(int)
            pbar = tqdm(input_dict.items())

            for connecting_user, connecting_dict in pbar:
                connected_users = [key for key, val in connecting_dict.items() if val >= mutual_limit and key in high_tweet_users]
                pairs = combinations(connected_users, 2)
                pbar.set_description("Mutual dict %s" % len(mutual_dict))
                for pair in pairs:
                    mutual_dict[frozenset(pair)] += 1

        else:

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

        for (connecting_user, connected_user), count in tqdm(mutual_dict.items()):
            if count >= connection_limit:
                if edge_weight:
                    graph.add_edge(connecting_user, connected_user, weight=count)
                else:
                    graph.add_edge(connecting_user, connected_user)

                graph.add_node(connecting_user, label=next(iter(user_dict[connecting_user])))
                graph.add_node(connected_user, label=next(iter(user_dict[connected_user])))

    return graph


# @mticker.FuncFormatter
# def major_formatter(x, pos):
#     label = str(-int(x)) if x < 0 else str(int(x))
#     return label


def load_csv_connection_data(link_csv, dict_pkl_file=None, users_pkl_file=None, node_csv=None):

    connections_dict = defaultdict(dict_dict)
    username_dict = dict()

    with open(link_csv, 'r') as readfile:
        reader = csv.DictReader(readfile)

        for row in tqdm(reader):
            source = int(row['user_id'])
            target = int(row['retweet_user_id'])

            if target not in connections_dict[source]:
                connections_dict[source][target] = 0
            connections_dict[source][target] += 1

            if node_csv is None:
                username_dict[int(source)] = set([str(source)])
                username_dict[int(target)] = set([str(target)])

    with open(dict_pkl_file, 'wb') as openfile:
        pickle.dump(connections_dict, openfile)
    with open(users_pkl_file, 'wb') as openfile:
        pickle.dump(username_dict, openfile)


def generate_shared_engagement_graphs(output_dir, network_dir, user_pkl, dict_pkl, prefix='whole_election', mutual_range=None, connection_range=None, overwrite=False):

    if connection_range is None:
        connection_range = [3, 4]

    network_file = f'{network_dir}/Election_Network_{prefix}'

    print('Loading user pkl')
    with open(user_pkl, 'rb') as openfile:
        user_pkl = pickle.load(openfile)

    print('Loading connections pkl')
    with open(dict_pkl, 'rb') as openfile:
        dict_pkl = pickle.load(openfile)

    for mutual_limit in range(*mutual_range):

        mutual_pkl = f'{output_dir}/{prefix}_mutual_connects_{mutual_limit}.pkl'
        if os.path.exists(mutual_pkl):
            with open(mutual_pkl, 'rb') as openfile:
                mutual_pkl = pickle.load(openfile)

        for connection_limit in range(*connection_range):

            output_filepath = f'{network_file}_mutual_{connection_limit}_{mutual_limit}_retweet.gexf'
            if not overwrite and os.path.exists(output_filepath):
                continue

            generate_network_gexf(
                database_name=None,
                db_config_file=None,
                input_json_dir=None,
                output_network_file=output_filepath,
                save_pkl=True,
                load_from_pkl=True,
                load_from_gexf=False,
                input_network_file=None,
                dict_pkl_file=dict_pkl,
                users_pkl_file=user_pkl,
                mutual_pkl_file=mutual_pkl,
                table_name='tweets',
                connection_type='retweet',
                link_type='mutual',
                conditions=[],
                attributes=None,
                label='screen_name',
                connection_limit=connection_limit,
                mutual_limit=mutual_limit,
                network_pruning=0,
                itersize=1000,
                limit=None,
                mode='networkx',
                overwrite=False,
                mutual_overwrite=False,
            )

    return


def n_filter_graph(input_graph, threshold=100, output_graph=None):

    print('Loading graph...')
    G = nx.read_gexf(input_graph)

    for edge in tqdm(list(G.edges())):
        if G.edges[edge]['weight'] < threshold:
            G.remove_edge(edge[0], edge[1])

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if output_graph is not None:
        nx.write_gexf(G, output_graph)

    return G


def count_nodes_in_clusters(input_graph, output_csv):

    print('Loading Graph..')
    G = nx.read_gexf(input_graph)

    cluster_dict = defaultdict(int)
    for node in G:
        cluster_dict[G.nodes[node]['Modularity Class']] += 1

    total = len(G.nodes)
    output_data = []
    print('Total', total)
    for key, item in cluster_dict.items():
        print(key, item, item / total)
        output_data += [[key, item, item / total]]

    df = pd.DataFrame(output_data, columns=['cluster', 'total', 'percent'])
    df.to_csv(output_csv, index=False)

    return


def cross_cluster_measurement(exemplar_network, output_data, cross_cutting_data, cluster_data, cumulative_data, only_clusters=None, weighted=False):

    print('Loading Graph')
    graph = nx.read_gexf(exemplar_network)

    # Tally cluster-to-cluster totals
    connections_dict = dict_dict()
    clusters = set()
    nodes = list(graph.nodes)
    for node in tqdm(nodes):

        if only_clusters is not None:
            if graph.nodes[node]['Modularity Class'] not in only_clusters:
                continue

        cluster = graph.nodes[node]['Modularity Class']

        connections_dict[node]['user_id'] = node
        connections_dict[node]['user_screen_name'] = graph.nodes[node]['label']
        connections_dict[node]['cluster'] = cluster
        connections_dict[node]['cross_connections'] = 0

        for edge in graph.edges(node):
            other_node = edge[1]
            other_cluster = graph.nodes[other_node]['Modularity Class']

            if only_clusters is not None:
                if other_cluster not in only_clusters:
                    continue
            clusters.add(other_cluster)

            if weighted:
                weight = graph[node][other_node]['weight']
            else:
                weight = 1

            if other_cluster in connections_dict[node]:
                connections_dict[node][other_cluster] += weight
            else:
                connections_dict[node][other_cluster] = weight

            if other_cluster != cluster:
                connections_dict[node]['cross_connections'] += weight

    # Count number of cross-cutting connections
    cross_dict = defaultdict(int)
    cross_dict_id = defaultdict(int)
    for key, item in connections_dict.items():
        for cluster in clusters:
            if cluster != item['cluster'] and cluster in item:
                cross_dict[item['user_screen_name']] += item[cluster]
                cross_dict_id[item['user_id']] += item[cluster]
    cross_dict_id = dict(sorted(cross_dict_id.items(), key=lambda item: item[1], reverse=True))

    # Get cumulative counts of cross-cutting connections, removing those from previous users iteratively
    cumulative_dict = defaultdict(int)
    cumulative_graph = deepcopy(graph)
    sorted_cross_ids = sorted(cross_dict_id, key=cross_dict_id.get, reverse=True)
    for user in sorted_cross_ids:
        user_edges = list(cumulative_graph.edges(user))
        for edge in user_edges:
            cluster = graph.nodes[user]['Modularity Class']
            other_cluster = graph.nodes[edge[1]]['Modularity Class']
            if cluster != other_cluster and cluster in only_clusters and other_cluster in only_clusters:
                cumulative_dict[graph.nodes[user]['label']] += 1
                cumulative_graph.remove_edge(edge[0], edge[1])

    # Tally individual cross-cluster totals.
    keynames = set()
    cluster_dict = defaultdict(int_dict)
    for cluster1, cluster2 in combinations(clusters, 2):
        keyname = '_'.join([str(cluster1), str(cluster2)])
        keynames.add(keyname)
        for key, item in connections_dict.items():
            if item['cluster'] == cluster1 and cluster2 in item:
                cluster_dict[item['user_screen_name']][keyname] += item[cluster2]
            if item['cluster'] == cluster2 and cluster1 in item:
                cluster_dict[item['user_screen_name']][keyname] += item[cluster1]
            cluster_dict[item['user_screen_name']]['total_edges'] = sum([item[cluster] for cluster in clusters if cluster in item])

    with open(output_data, 'w') as writefile, open(cross_cutting_data, 'w') as crossfile, \
            open(cluster_data, 'w') as clusterfile, open(cumulative_data, 'w') as cumfile:

        header = ['user_id', 'user_screen_name', 'cluster', 'cross_connections'] + list(clusters)
        writer = csv.DictWriter(writefile, fieldnames=header)
        writer.writeheader()

        for key, item in connections_dict.items():
            writer.writerow(item)

        header = ['user_screen_name', 'cross_cuts']
        writer = csv.writer(crossfile)
        writer.writerow(header)

        for key, item in cross_dict.items():
            writer.writerow([key, item])

        header = ['user_screen_name', 'cumulative_cross_cuts']
        writer = csv.writer(cumfile)
        writer.writerow(header)

        for key, item in cumulative_dict.items():
            writer.writerow([key, item])

        header = ['user_screen_name', 'total_edges'] + list(keynames)
        writer = csv.DictWriter(clusterfile, fieldnames=header)
        writer.writeheader()

        for key, item in cluster_dict.items():
            item['user_screen_name'] = key
            writer.writerow(item)

    return


def extend_follower_data(reference_graph, input_csv, output_csv):

    df = pd.read_csv(input_csv, index_col=0)
    print(df)

    graph = nx.read_gexf(reference_graph)
    data = []
    for node in graph.nodes:
        data += [[int(node), graph.nodes[node]['Modularity Class']]]
    cluster_df = pd.DataFrame(data, columns=['user_id', 'cluster'])

    df = pd.merge(df, cluster_df, 'left', 'user_id')
    df.to_csv(output_csv, index=False)

    return


def suspension_statistics(input_graphs, output_dir, suspended_ids, output_csv):

    inactive_ids = pd.read_csv(suspended_ids)
    inactive_ids = list(inactive_ids['user_id'])

    with open(output_csv, 'w') as writefile:

        writer = csv.writer(writefile, delimiter=',')
        writer.writerow(['Case', 'Cluster', 'Total', 'Removed', 'Percent', 'Edges Total', 'Edges Removed', 'Edges Percent'])

        for key, input_graph in input_graphs.items():

            print('Loading... ', key)
            output_graph = os.path.join(output_dir, 'Suspended_' + os.path.basename(input_graph))

            tree = ET.parse(input_graph)
            root = tree.getroot()

            viz_dict = {}
            for node_info in root[1][1]:
                viz_dict[node_info.attrib['id']] = {
                    'size': float(node_info[1].attrib['value']),
                    'position': {
                        'x': float(node_info[2].attrib['x']),
                        'y': float(node_info[2].attrib['y']),
                        'z': 0,
                    },
                    'color': {
                        'r': int(node_info[0].attrib['r']),
                        'g': int(node_info[0].attrib['g']),
                        'b': int(node_info[0].attrib['b']),
                        'a': 1,
                    }
                }

            graph = nx.read_gexf(input_graph)

            removed_dict = defaultdict(int)
            counts = defaultdict(int)
            nodes = list(graph.nodes)
            for node in tqdm(nodes):

                graph.nodes[node]['viz'] = viz_dict[node]

                counts[graph.nodes[node]['Modularity Class']] += 1

                if int(node) in inactive_ids:
                    removed_dict[graph.nodes[node]['Modularity Class']] += 1
                    graph.nodes[node]['removed'] = 1
                    graph.nodes[node]['viz']['color'] = {'r': 0, 'b': 0, 'g': 0, 'a': 1}
                    for edge in graph.edges(node):
                        graph[edge[0]][edge[1]]['removed'] = 1
                else:
                    graph.nodes[node]['removed'] = 0
                    for edge in graph.edges(node):
                        graph[edge[0]][edge[1]]['removed'] = 0

            removed_edge_dict = defaultdict(int)
            edge_counts = defaultdict(int)
            edges = list(graph.edges)
            for node1, node2 in tqdm(edges):

                for node in node1, node2:
                    edge_counts[graph.nodes[node]['Modularity Class']] += 1
                    if node1 == node2:
                        break

                if any([int(x) in inactive_ids for x in [node1, node2]]):
                    removed_edge_dict[graph.nodes[node]['Modularity Class']] += 1
                    graph[node1][node2]['removed'] = 1
                else:
                    graph[node1][node2]['removed'] = 0

            nx.write_gexf(graph, output_graph)

            pprint(counts)

            for val, count in counts.items():
                writer.writerow([key, val, count, removed_dict[val], removed_dict[val] / count, edge_counts[val], removed_edge_dict[val], removed_edge_dict[val] / edge_counts[val]])

    return


def s_1_graph(dict_pkl, user_pkl, output_dir, network_dir, prefix='whole_election', connection_limit=10000, overwrite=False):

    skip_mutual = True
    mutual_limit = 1
    network_file = f'{network_dir}/Election_Network_{prefix}'

    print('Loading user pkl')
    with open(user_pkl, 'rb') as openfile:
        user_pkl = pickle.load(openfile)

    print('Loading connections pkl')
    with open(dict_pkl, 'rb') as openfile:
        dict_pkl = pickle.load(openfile)

    mutual_pkl = f'{output_dir}/{prefix}_mutual_connects_{mutual_limit}.pkl'

    output_filepath = f'{network_file}_mutual_{connection_limit}_{mutual_limit}_retweet.gexf'
    if not overwrite and os.path.exists(output_filepath):
        return

    generate_network_gexf(
        database_name=None,
        db_config_file=None,
        input_json_dir=None,
        output_network_file=output_filepath,
        save_pkl=True,
        load_from_pkl=True,
        load_from_gexf=False,
        input_network_file=None,
        dict_pkl_file=dict_pkl,
        users_pkl_file=user_pkl,
        mutual_pkl_file=mutual_pkl,
        table_name='tweets',
        connection_type='retweet',
        link_type='mutual',
        conditions=[],
        attributes=None,
        label='screen_name',
        connection_limit=connection_limit,
        mutual_limit=mutual_limit,
        network_pruning=0,
        itersize=1000,
        limit=None,
        mode='networkx',
        overwrite=False,
        mutual_overwrite=False,
        skip_mutual=skip_mutual,
    )

    return


def get_linked_users(input_csv, input_network, output_csv):

    users = list(pd.read_csv(input_csv)['user_screen_name'])

    graph = nx.read_gexf(input_network)

    output_data = []
    for node in tqdm(users):
        neighbors = graph.neighbors(str(node))
        for neighbor in neighbors:
            output_data += [[neighbor, node, graph.nodes[neighbor]['label'], node, graph.nodes[neighbor]['Modularity Class']]]

    df = pd.DataFrame(output_data, columns=['user_id', 'linking_users', 'user_screen_name', 'linking_user_screen_name', 'cluster'])
    df.to_csv(output_csv, index=False)

    return


def generate_linked_user_data(input_csv, dict_pkl, output_pkl, input_graph):

    user_ids = [int(x) for x in nx.read_gexf(input_graph).nodes]

    output_dict = defaultdict(list)

    print('Loading connections pkl')
    with open(dict_pkl, 'rb') as openfile:
        dict_pkl = pickle.load(openfile)

    pbar = tqdm(dict_pkl.items())
    for connecting_user, connecting_dict in pbar:
        for connected_user, val in connecting_dict.items():
            if connected_user in user_ids:
                output_dict[connected_user] += [connecting_user]

    with open(output_pkl, 'wb') as openfile:
        pickle.dump(output_dict, openfile)

    return


def extract_linked_tweets(raw_data, linking_csv, output_csv):

    print('Loading linking csv')
    linking_df = pd.read_csv(linking_csv)
    linking_users = [str(x) for x in set(linking_df['linking_users'])]

    with open(raw_data, 'r') as readfile, open(output_csv, 'w') as writefile:
        reader = csv.DictReader(readfile)
        writer = csv.writer(writefile)
        writer.writerow(['retweeted_status_id', 'user_id', 'retweeted_status_user_id', 'created_at'])

        for row in tqdm(reader):
            if row['retweet_user_id'] in linking_users:
                writer.writerow([row['retweet_id'], row['user_id'], row['retweet_user_id'], row['created_at']])

    return


def combine_reduce_tweets(input_csv, input_pkl, linking_csv, output_csv, linking_graph, left_cluster=1):

    print('Loading linking csv')
    linking_df = pd.read_csv(linking_csv)
    linking_users = list(set(linking_df['linking_users']))

    print('Loading bridging connection data pkl')
    with open(input_pkl, 'rb') as openfile:
        user_data = pickle.load(openfile)

    print('Loading graph')
    link_graph = nx.read_gexf(linking_graph)

    connect_data = []
    for node in tqdm(list(link_graph.nodes)):
        if int(node) in linking_users:
            continue
        user_connections = user_data[int(node)]
        for connect_user in user_connections:
            if link_graph.nodes[node]['Modularity Class'] == left_cluster:
                connect_data += [[connect_user, True, False]]
            else:
                connect_data += [[connect_user, False, True]]

    connect_data = pd.DataFrame(connect_data, columns=['user_id', 'left', 'right']).groupby('user_id').agg({'left': 'max', 'right': 'max'}).reset_index()
    connect_data['both'] = False
    connect_data.loc[connect_data['right'] & connect_data['left'], 'both'] = True

    print('Loading tweet data')
    dtypes = {
        'user_id': 'int64',
        'created_at': 'object',
        'retweeted_status_id': 'int64',
        'retweeted_status_user_id': 'int64',
    }
    parse_dates = ['created_at']
    df = pd.read_csv(input_csv, parse_dates=parse_dates, dtype=dtypes).drop_duplicates()

    output_df = None
    for link_user in tqdm(linking_users):
        subdf = df[df['retweeted_status_user_id'] == link_user]
        result_df = pd.merge(connect_data, subdf, 'left', 'user_id')

        group_df = result_df.groupby('retweeted_status_id').agg({'user_id': 'count', 'retweeted_status_user_id': 'first', 'created_at': 'first', 'left': 'sum', 'right': 'sum', 'both': 'sum'})

        if output_df is None:
            output_df = group_df
        else:
            output_df = pd.concat([output_df, group_df])

    output_df.to_csv(output_csv)

    return


def graph_tweet_allignment(input_csv, plot_prefix, min_tweets=1, show_plot=False):

    df = pd.read_csv(input_csv)
    df['total'] = df['left'] + df['right']
    df['left'] = -1 * df['left']
    df = df[df['total'] >= min_tweets]
    df['created_at'] = pd.to_datetime(df['created_at'])
    link_users = list(pd.unique(df['retweeted_status_user_id']))
    print(link_users)
    print(df)

    for link_user in link_users:
        link_df = df[df['retweeted_status_user_id'] == link_user].sort_values('created_at')
        print(link_df)
        new_df = []
        for idx, row in link_df.iterrows():
            new_df += [[row['created_at'] + pd.Timedelta(seconds=-1), 0, 0]]
            new_df += [[row['created_at'], row['left'] + row['both'], row['right'] - row['both']]]
            new_df += [[row['created_at'] + pd.Timedelta(seconds=1), 0, 0]]

        new_df = pd.DataFrame(new_df, columns=['Date', 'Pro-Biden Only', 'Pro-Trump Only'])
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        new_df = new_df.sort_values('Date')
        new_df.set_index(['Date'], inplace=True)

        ax = new_df.plot(
            x_compat=True, style={'Pro-Biden Only': ',-b', 'Pro-Trump Only': ',-r'},
            title=f'Partisan Retweets of @{link_user}', grid=True, figsize=(16, 4),
        )

        ax.yaxis.set_major_formatter(major_formatter)
        max_ylim = max([abs(x) for x in ax.get_ylim()])
        ax.set_ylim([-1 * max_ylim, max_ylim])
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        plt.savefig(f'{plot_prefix}_{link_user}.png', bbox_inches='tight')

        if show_plot:
            plt.show()


def characterize_self_links(dict_pkl, input_graph, clusters, output_csv, connection_limit=25):

    graph = nx.read_gexf(input_graph)
    print('Loading connections pkl')
    with open(dict_pkl, 'rb') as openfile:
        dict_pkl = pickle.load(openfile)

    data = []
    for cluster in clusters:
        cluster_nodes = [int(x) for x in graph.nodes if graph.nodes[x]['Modularity Class'] == cluster]

        in_total = 0
        out_total = 0
        pbar = tqdm(dict_pkl.items())
        for connecting_user, connecting_dict in pbar:
            for connected_user, val in connecting_dict.items():
                if val < connection_limit:
                    continue
                if connected_user in cluster_nodes:
                    if connecting_user in cluster_nodes:
                        in_total += 1
                    else:
                        out_total += 1

        output_data = [cluster, in_total, out_total, in_total / (in_total + out_total)]
        print(output_data)
        data += [output_data]

    output_df = pd.DataFrame(data, columns=['cluster', 'self_links', 'other_links', 'self_link_ratio'])
    output_df.to_csv(output_csv, index=False)

    return


def cluster_graphs_volume_threshold(input_dir, color_ref, output_cluster_csv, s_min=5, bootstrap_iterations=3):

    graphs = sorted(glob(os.path.join(input_dir, '*mutual_3_*.gexf')))
    account_ref = pd.read_csv(color_ref, delimiter=':', header=None, names=['label', 'color', 'cluster'])
    clusters = list(pd.unique(account_ref['cluster']))

    with open(output_cluster_csv, 'w') as writefile:
        writer = csv.writer(writefile, delimiter=',')
        writer.writerow(['n', 's', 'cluster', 'count'])

        pbar = tqdm(graphs, desc='Bootstrap:')
        for graph in pbar:

            print(graph)
            split_graph = str.split(os.path.basename(graph), '_')
            s = int(split_graph[-2])

            if s < s_min:
                continue

            G = nx.read_gexf(graph)

            n_set = set()
            for edge in G.edges():
                n_set.add(G.edges[edge]['weight'])
            n_set = list(sorted(n_set))

            cluster_dict = {}
            for idx, row in account_ref.iterrows():
                if row['label'] in G.nodes:
                    cluster_dict[row['label']] = row['cluster']

            for n in tqdm(n_set):
                for edge in list(G.edges()):
                    if G.edges[edge]['weight'] < n:
                        G.remove_edge(edge[0], edge[1])

                isolates = list(nx.isolates(G))
                G.remove_nodes_from(isolates)

                bootstrap_list = []
                for bootstrap in range(bootstrap_iterations):
                    partition = community_louvain.best_partition(G)

                    output_dict = {label: 0 for label in clusters}
                    classify_dict = {}
                    for lead_cluster in reversed(clusters):
                        cluster_vals = {key: val for key, val in cluster_dict.items() if val == lead_cluster}
                        for node_id, cluster in cluster_vals.items():
                            if node_id in G.nodes:
                                cluster_id = partition[node_id]
                                for subnode, subid in partition.items():
                                    if subid == cluster_id:
                                        classify_dict[subnode] = cluster

                    for node, cluster in classify_dict.items():
                        output_dict[cluster] += 1

                    pbar.set_description("Bootstrap: %s" % bootstrap)

                    bootstrap_list += [output_dict]

                    for key, item in output_dict.items():
                        writer.writerow([n, s, key, item])

                output_dict_mean = {}
                output_dict_median = {}
                for cluster in clusters:
                    output_dict_mean[cluster] = float(np.mean([subdict[cluster] for subdict in bootstrap_list]))
                    output_dict_median[cluster] = float(np.median([subdict[cluster] for subdict in bootstrap_list]))

    return


def visualize_cluster_boundaries(input_csv, output_viz_filepath):

    df = pd.read_csv(input_csv).sort_values('n')
    clusters = pd.unique(df['cluster'])

    df = df.groupby(['n', 's', 'cluster'], as_index=False).median().sort_values('n')

    array_dict = {}
    for cluster in tqdm(clusters):
        cluster_df = df[df['cluster'] == cluster].sort_values(['s', 'n'])
        array = np.zeros((int(cluster_df['n'].max()) + 1, int(cluster_df['s'].max()) + 1), dtype=float)
        prev_n = 3
        for idx, row in cluster_df.iterrows():
            if row['count'] > 0:
                array[prev_n:int(row['n']), int(row['s'])] = 1
            prev_n = int(row['n'])
        array_dict[cluster] = array

    plt.figure(figsize=(16, 12), dpi=80)

    font = {
        'family': 'normal',
        'size': 18,
    }

    plt.rc('font', **font)

    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ])

    color_dict = {
        'right': ['lightcoral', 'red'],
        'left': ['navy', 'royalblue'],
        'rose': ['olivedrab', 'green'],
        'trumptrain': ['lightsalmon', 'orangered'],
        'bluewave': ['deepskyblue', 'skyblue'],
        'italian': ['yellow', 'khaki'],
    }

    cluster_name_dict = {
        'right': 'Pro-Trump',
        'left': 'Pro-Biden',
        'rose': 'Pro-socialist',
        'trumptrain': 'Pro-Trump Followback',
        'bluewave': 'Pro-Biden Followback',
        'italian': '"Trump\'s Italians"',
    }

    cluster_order = ['right', 'left', 'trumptrain', 'italian', 'rose', 'bluewave']
    graph_dict = {}
    for cluster in cluster_order:
        print(cluster)
        cluster_array = array_dict[cluster]

        print('Finding boundary')
        padded = np.pad(cluster_array, ((1, 1), (1, 1)), 'constant', constant_values=0)
        frontier = convolve2d(1 - padded, kernel, mode='valid').astype(bool) * cluster_array

        print('Cutting outline')
        x, y = frontier.nonzero()
        newx, newy = [], []
        minx, maxx = max(3, min(x)), max(x)

        for idx in range(minx, maxx + 1):
            if idx in x:
                newx += [idx]
                newy += [min(y[x == idx])]

        for idx in reversed(range(minx - 1, maxx)):
            if idx in x:
                newx += [idx]
                newy += [max(y[x == idx])]

        graph_dict[cluster] = [newx, newy]

    for cluster in cluster_order:
        newx, newy = graph_dict[cluster]
        print('Drawing')
        plt.fill(newx, newy, facecolor=color_dict[cluster][0], edgecolor=color_dict[cluster][1], linewidth=5, alpha=.25, label=cluster_name_dict[cluster])

    ax = plt.gca()
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("s")
    plt.title("Cluster Membership in n/s Parameter Space")
    plt.xscale('log')
    plt.xlim([5, 130000])
    plt.ylim([4, 300])
    # plt.yscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    xticks = [5, 10, 100, 1000, 10000, 100000]
    ax.set_xticks(xticks)
    yticks = [4] + list(range(50, 350, 50))
    ax.set_yticks(yticks)

    plt.savefig(output_viz_filepath, bbox_inches='tight')

    return


def dataset_statistics(user_pkl, dict_pkl, output_csv):

    with open(output_csv, 'w') as writefile:
        writer = csv.writer(writefile)

        print('Loading users pkl')
        with open(user_pkl, 'rb') as openfile:
            user_pkl = pickle.load(openfile)

        total_users = len(user_pkl)
        print('Total users', total_users)
        writer.writerow(['Total_Users', total_users])

        print('Loading connections pkl')
        with open(dict_pkl, 'rb') as openfile:
            dict_pkl = pickle.load(openfile)

        total_tweets = 0
        total_relations = 0
        pbar = tqdm(dict_pkl.items())
        for connecting_user, connecting_dict in pbar:
            for connected_user, val in connecting_dict.items():
                total_tweets += val
                total_relations += 1

        print('Total Retweets', total_tweets)
        writer.writerow(['Total_Retweets', total_tweets])

        print('Total Relations', total_relations)
        writer.writerow(['Total_Relations', total_relations])

    return


# Revisions

def convert_pkl_to_csv(dict_pkl, output_csv):

    print('Loading connections pkl')
    with open(dict_pkl, 'rb') as openfile:
        dict_pkl = pickle.load(openfile)

    pbar = tqdm(dict_pkl.items())
    with open(output_csv, 'w') as writefile:
        writer = csv.writer(writefile)

        for connecting_user, connecting_dict in pbar:
            for connected_user, val in connecting_dict.items():
                writer.writerow([connecting_user, connected_user, val])

    return


def create_directed_retweet_gexf(dict_pkl, output_npz, output_nodelist):

    print('Loading connections pkl')
    with open(dict_pkl, 'rb') as openfile:
        dict_pkl = pickle.load(openfile)

    output_graph = nx.DiGraph()
    pbar = tqdm(dict_pkl.items())
    for connecting_user, connecting_dict in pbar:
        for connected_user, val in connecting_dict.items():
            output_graph.add_edge(connecting_user, connected_user, weight=val)

    del connecting_dict

    nodelist = output_graph.nodes()
    output_matrix = nx.to_scipy_sparse_matrix(output_graph, nodelist=nodelist)
    save_npz(output_npz, output_matrix)

    with open(output_nodelist, 'wb') as fp:
        pickle.dump(nodelist, fp)

    print(output_matrix)

    return


def create_directed_retweet_gexf_2(input_csv, output_npz, output_nodelist):

    output_graph = nx.DiGraph()
    with open(input_csv, 'r') as readfile:
        reader = csv.reader(readfile)
        for row in tqdm(reader):
            output_graph.add_edge(int(row[0]), int(row[1]), weight=int(row[2]))

    nodelist = output_graph.nodes()
    output_matrix = nx.to_scipy_sparse_matrix(output_graph, nodelist=nodelist)
    save_npz(output_npz, output_matrix)

    with open(output_nodelist, 'wb') as fp:
        pickle.dump(nodelist, fp)

    print(output_matrix)

    return


def cluster_directed_retweet_gexf(input_gexf, output_partitions):

    # input_matrix = load_npz(input_npz)
    # print(input_matrix)
    # graph = nx.from_scipy_sparse_matrix(input_matrix)
    # print(graph)
    
    print('Loading Graph')
    graph = nx.read_gexf(input_gexf)

    print('Converting to Undirected')
    ugraph = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if ugraph.has_edge(u, v):
            if ugraph[u][v]['weight'] < data['weight']:
                ugraph[u][v]['weight'] = data['weight']
        else:
            ugraph.add_edge(u, v, weight=data['weight'])

    print(graph)
    print(ugraph)

    print('Partitioning Graph')
    partition = community_louvain.best_partition(ugraph)

    with open(output_partitions, 'wb') as fp:
        pickle.dump(partition, fp)

    return


def get_indegree_list_csv(input_csv, user_pkl, output_csv):

    degree_dict = defaultdict(int)
    with open(input_csv, 'r') as readfile:
        reader = csv.reader(readfile)
        for row in tqdm(reader):
            degree_dict[int(row[1])] += int(row[2])

    print('Loading user data')
    with open(user_pkl, 'rb') as openfile:
        user_dict = pickle.load(openfile)

    output_data = []
    for key, item in degree_dict.items():
        try:
            output_data += [[key, item, next(iter(user_dict[key]['screen_name']))]]
        except:
            output_data += [[key, item, '']]

    df = pd.DataFrame(output_data, columns=['id', 'indegree', 'screen_name'])
    df.to_csv(output_csv, index=False)

    return


def get_cluster_distribution_infomap(indegree_list, input_clusters, case, output_csv, top=1000):

    dtypes = {
        'path': 'object',
        'flow': 'float',
        'name': 'int64',
        'node_id': 'int64',
        'cluster': 'int64',
        'indegree': 'int64',
        'id': 'int64',
        'screen_name': 'object'
    }
    
    print('loading graph')
    graph = nx.read_gexf(case)
    cluster_dict = defaultdict(list)
    for node in graph.nodes:
        cluster_dict[graph.nodes[node]['Modularity Class']] += [int(node)]
    nodes = [int(x) for x in graph.nodes]

    print('loading data')
    df_infomap = pd.read_csv(input_clusters, dtype=dtypes).sort_values('cluster')
    print(df_infomap) 
    df_infomap = df_infomap[df_infomap['name'].isin(nodes)]
    print(df_infomap)  

    output_df = pd.DataFrame([x for x in list(df_infomap.value_counts('cluster').index)], columns=['cluster'])
    for key, nodes in cluster_dict.items():
        cluster_df = df_infomap[df_infomap['name'].isin(nodes)]
        cluster_counts = cluster_df.value_counts('cluster')
        
        cluster_counts = [[x, y] for x, y in cluster_counts.iteritems()]
        cluster_counts = pd.DataFrame(cluster_counts, columns=['cluster', key])
        output_df = pd.merge(output_df, cluster_counts, 'left', 'cluster')

    output_df.to_csv(output_csv, index=False)


def get_highest_degree_clusters(input_clusters, indegree_list, output_csv, top=100):

    dtypes = {
        'path': 'object',
        'flow': 'float',
        'name': 'int64',
        'node_id': 'int64',
        'cluster': 'int64',
        'indegree': 'int64',
        'id': 'int64',
        'screen_name': 'object'
    }
    
    print('loading data')
    df_infomap = pd.read_csv(input_clusters, dtype=dtypes)

    top_100_clusters = df_infomap['cluster'].value_counts().head(top)
    top_100_clusters = list(top_100_clusters.index)
    print(top_100_clusters)
    df_infomap = df_infomap[df_infomap['cluster'].isin(top_100_clusters)]
    print(df_infomap)

    df_indegree = pd.read_csv(indegree_list, dtype=dtypes)
    df_indegree = df_indegree.rename(columns={"id": "name"})
    print(df_indegree)

    df_infomap = pd.merge(df_infomap, df_indegree, 'left', 'name')
    print(df_infomap)

    df_infomap.to_csv(output_csv)

    return


def get_highest_degree_nodes(input_clusters, indegree_list, output_csv, top=1000):

    dtypes = {
        'path': 'object',
        'flow': 'float',
        'name': 'int64',
        'node_id': 'int64',
        'cluster': 'int64',
        'indegree': 'int64',
        'id': 'int64',
        'screen_name': 'object'
    }
    
    print('loading data')
    df_infomap = pd.read_csv(input_clusters, dtype=dtypes)
    df_indegree = pd.read_csv(indegree_list, dtype=dtypes)
    df_indegree = df_indegree.rename(columns={"id": "name"})
    print(df_indegree)

    df_infomap = pd.merge(df_infomap, df_indegree, 'left', 'name')
    print(df_infomap)

    df_infomap = df_infomap.sort_values('indegree', ascending=False).head(1000)
    print(df_infomap)

    df_infomap.to_csv(output_csv)

    return


def get_percent_of_top_1000(indegree_csv, input_nodes, exclude_clusters=None):

    dtypes = {
        'path': 'object',
        'flow': 'float',
        'name': 'int64',
        'node_id': 'int64',
        'cluster': 'int64',
        'indegree': 'int64',
        'id': 'int64',
        'screen_name': 'object'
    }

    print('loading graph')
    nodes = [int(x) for x in nx.read_gexf(input_nodes).nodes()]

    df_indegree = pd.read_csv(indegree_csv, dtype=dtypes)
    df_indegree = df_indegree.rename(columns={"id": "name"}).sort_values('indegree', ascending=False).head(1000)
    print(df_indegree)

    df_indegree_case = df_indegree[df_indegree['name'].isin(nodes)]
    print(df_indegree_case)
    # print(df_indegree_case['indegree'].sum(), df_indegree['indegree'].sum(), df_indegree_case['indegree'].sum() / df_indegree['indegree'].sum())    


def get_cluster_percentage_of_total(indegree_csv, input_clusters, input_nodes, exclude_clusters=None):

    dtypes = {
        'path': 'object',
        'flow': 'float',
        'name': 'int64',
        'node_id': 'int64',
        'cluster': 'int64',
        'indegree': 'int64',
        'id': 'int64',
        'screen_name': 'object'
    }

    print('loading graph')
    nodes = [int(x) for x in nx.read_gexf(input_nodes).nodes()]

    print('loading data')
    df_infomap = pd.read_csv(input_clusters, dtype=dtypes)

    # top_100_clusters = df_infomap['cluster'].value_counts().head(100)
    # top_100_clusters = list(top_100_clusters.index)
    # df_infomap = df_infomap[df_infomap['cluster'].isin(top_100_clusters)]
    # print(df_infomap)

    df_infomap = df_infomap[~df_infomap['name'].isin(nodes)]

    df_indegree = pd.read_csv(indegree_csv, dtype=dtypes)
    df_indegree = df_indegree.rename(columns={"id": "name"})
    print(df_indegree)

    df_indegree_case = df_indegree[df_indegree['name'].isin(nodes)]
    print(df_indegree_case['indegree'].sum(), df_indegree['indegree'].sum(), df_indegree_case['indegree'].sum() / df_indegree['indegree'].sum())    

    df_infomap = pd.merge(df_infomap, df_indegree, 'left', 'name').sort_values('indegree', ascending=False).head(20)
    print(df_infomap)

    return


def isolate_cluster(cluster_csv, network_csv, output_graph, cluster=17):

    dtypes = {
        'path': 'object',
        'flow': 'float',
        'name': 'int64',
        'node_id': 'int64',
        'cluster': 'int64',
        'indegree': 'int64',
        'id': 'int64',
        'screen_name': 'object'
    }
    
    print('loading data')
    df_infomap = pd.read_csv(cluster_csv, dtype=dtypes)  
    cluster_ids = list(df_infomap[df_infomap['cluster'] == cluster]['name'])
    print(len(cluster_ids))
    print(type(cluster_ids[0]))

    graph = nx.DiGraph()

    with open(network_csv, 'r') as readfile:
        reader = csv.reader(readfile)

        for row in tqdm(reader):
            if int(row[1]) in cluster_ids:
                graph.add_edge(int(row[0]), int(row[1]), weight=int(row[2]))
                continue
            if int(row[0]) in cluster_ids:
                graph.add_edge(int(row[0]), int(row[1]), weight=int(row[2]))

    nx.write_gexf(graph, output_graph)

    return


def relabel_graph(input_graph, output_graph, user_pkl):

    print('Loading graph')
    graph = nx.read_gexf(input_graph)

    print('Loading user data')
    with open(user_pkl, 'rb') as openfile:
        user_dict = pickle.load(openfile)

    for node in tqdm(list(graph.nodes)):
        graph.nodes[node]['label'] = next(iter(user_dict[int(node)]['screen_name']))

    nx.write_gexf(graph, output_graph)

    return


def relabel_highest_clusters(input_csv, user_pkl, output_csv):

    print('Loading Cluster data')
    data = pd.read_csv(input_csv)

    print('Loading user data')
    with open(user_pkl, 'rb') as openfile:
        user_dict = pickle.load(openfile)

    # print(user_dict)

    output_data = []
    for idx, row in tqdm(list(data.iterrows())):
        try:
            output_data += [list(row) + [next(iter(user_dict[int(row['id'])]['screen_name']))]]
        except:
            continue

    df = pd.DataFrame(output_data, columns=['id', 'cluster', 'degree', 'weighted_degree', 'label'])
    df.to_csv(output_csv, index=False)


def load_infomap_clusters(input_tree, output_csv):

    dtypes = {
        'path': 'object',
        'flow': 'float',
        'name': 'int64',
        'node_id': 'int64',
    }
    
    print('loading data')
    df_infomap = pd.read_csv(input_tree, sep=" ", comment="#", names="path flow name node_id".split(), usecols=['path', 'name'])
    print('loading cluster')
    df_infomap['cluster'] = df_infomap.path.map(lambda x: x.split(':')[0]).astype(int)
    df_infomap['path'] = None

    print(df_infomap)

    vcs = df_infomap['cluster'].value_counts()
    print(vcs.head(25))

    df_infomap.to_csv(output_csv, index=False)

    return


def download_raw_tweets(output_filepath, overwrite=False):

    download_link = 'https://drive.google.com/uc?id=1pIjKj-QNh1VXzEbzRa3C-rvKT1k2DDy3'
    # download_link = 'https://drive.google.com/uc?id=1ihT6mf6l8YAVtNqQHdk-t19XlMmYLQ16'
    if not os.path.exists(output_filepath) or overwrite:
        gdown.download(download_link, output_filepath, quiet=False)

    # Don't know why I can't get this to work right.
    untarred = os.path.join(os.path.dirname(output_filepath), '../Datasets/Final_Draft_Graph/Final_Paper_Statistics_Figures/Open_Source_Data.csv')
    move_path = os.path.join(os.path.dirname(output_filepath), 'Open_Source_Data.csv')
    if not (os.path.exists(untarred) or os.path.exists(move_path)) or overwrite:
        with tarfile.open(output_filepath, 'r') as t:
            t.extractall(os.path.dirname(output_filepath))

    if not os.path.exists(move_path) or overwrite:
        shutil.move(untarred, move_path)

    if os.path.exists(os.path.join(os.path.dirname(output_filepath)), '../Datasets'):
        shutil.rmtree(os.path.join(os.path.dirname(output_filepath)), '../Datasets')

    # shutil.unpack_archive(output_filepath, os.path.dirname(output_filepath))

    # print(output_filepath)
    # with ZipFile(output_filepath, 'r') as zip_ref:
    #     zip_ref.extractall(os.path.dirname(output_filepath))

    # unzipped = output_filepath[:-3] + '.csv'
    # with lzma.open(output_filepath, 'r') as f, open(unzipped, 'w') as w:
    #     w.write(f.read())

    return


def download_dataset(dict_pkl, user_pkl, overwrite=False):

    dict_pkl_download_link = 'https://drive.google.com/uc?id=1jKkmgRtlBDY33Ug8paK9S0j2Xp0ebAA2'
    user_pkl_download_link = 'https://drive.google.com/uc?id=1Rt5mlEu-vJRwMyr0n2icFdLVawXJ1Rdy'

    if not os.path.exists(dict_pkl) or overwrite:
        gdown.download(dict_pkl_download_link, dict_pkl, quiet=False)
    if not os.path.exists(user_pkl) or overwrite:
        gdown.download(user_pkl_download_link, user_pkl, quiet=False)

    return


def download_case_study(case_study, output_filepath, overwrite=False):

    case_study_dict = {
        'Case 1': 'https://drive.google.com/uc?id=1zjOn1EJasK_LyIZCDQprLIwxMoUinsKB',
        'Case 2': 'https://drive.google.com/uc?id=1sEGkzqXVqXuWisaX5OzpQ2etxFPslERB',
        'Case 3': 'https://drive.google.com/uc?id=1LIm10D0DOmhgg7V0ou-v5Yusq7hRqvGe',
    }

    url = case_study_dict[case_study]
    if not os.path.exists(output_filepath) or overwrite:
        gdown.download(url, output_filepath, quiet=False)

    return


def download_bridge_user_data(tweet_filepath, bridge_connects_filepath, overwrite=False):

    download_link = 'https://drive.google.com/uc?id=19INk-kTOucY1-XBrVMVO8Lalf1J6om9j'
    if not os.path.exists(tweet_filepath) or overwrite:
        gdown.download(download_link, tweet_filepath, quiet=False)

    download_link = 'https://drive.google.com/uc?id=1ukvO6L5cMj3TLU_svuJfhQVsuw2TWOgy'
    if not os.path.exists(bridge_connects_filepath) or overwrite:
        gdown.download(download_link, bridge_connects_filepath, quiet=False)

    return


def download_follower_data(output_filepath, overwrite=False):

    download_link = 'https://drive.google.com/uc?id=1HMo8sU5aQJVlmdF-W8G6Oy6gUKYs906n'
    if not os.path.exists(output_filepath) or overwrite:
        gdown.download(download_link, output_filepath, quiet=False)

    return


def download_clustering_reference(output_filepath, overwrite=False):

    download_link = 'https://drive.google.com/uc?id=15zIqCydQpUMGtT9EhylLaXhlWppnOSlp'
    if not os.path.exists(output_filepath) or overwrite:
        gdown.download(download_link, output_filepath, quiet=False)

    return


def download_suspensions(output_filepath, overwrite=False):

    download_link = 'https://drive.google.com/uc?id=1ZSRKbc7YH7icRzcWtxp2vLI_NG8WXwgv'
    if not os.path.exists(output_filepath) or overwrite:
        gdown.download(download_link, output_filepath, quiet=False)

    return


def download_cluster_boundaries(output_filepath, overwrite=False):

    download_link = 'https://drive.google.com/uc?id=1hGvpup0baVHxmi_lCEl3JhbR48C3ufZK'
    if not os.path.exists(output_filepath) or overwrite:
        gdown.download(download_link, output_filepath, quiet=False)

    return


if __name__ == '__main__':

    # Create necessary folders
    if True:
        for folder in ['Intermediate_Files', 'Parameter_Space', 'Final_Paper_Statistics_Figures', 'Case_Study_Graphs', 'Comparison_Graph', 'Suspension_Graphs']:
            if not os.path.exists(os.path.join(f'../Datasets/{folder}')):
                os.mkdir(os.path.join(f'../Datasets/{folder}'))

    # Revisions
    if True:

        # Dataset tweet and user totals.
        if False:
            user_pkl = '../Datasets/Intermediate_Files/whole_election_user_connects_restricted_terms_anon.pkl'
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            dataset_stats_csv = '../Datasets/Final_Paper_Statistics_Figures/dataset_stats_overall.csv'
            dataset_statistics(user_pkl, dict_pkl, dataset_stats_csv)

        # Generate Direct Graph for Comparison Section
        if False:
            user_pkl = '../Datasets/Intermediate_Files/whole_election_user_connects_restricted_terms_anon.pkl'
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            output_filepath = '../Datasets/Comparison_Graph/Election_Network_whole_election_Direct_Retweet_2.gexf'
            generate_network_gexf(
                database_name=None,
                db_config_file=None,
                input_json_dir=None,
                output_network_file=output_filepath,
                save_pkl=True,
                load_from_pkl=True,
                load_from_gexf=False,
                input_network_file=None,
                dict_pkl_file=dict_pkl,
                users_pkl_file=user_pkl,
                mutual_pkl_file=None,
                table_name='tweets',
                connection_type='retweet',
                link_type='direct',
                conditions=[],
                attributes=None,
                label='screen_name',
                connection_limit=2,
                mutual_limit=0,
                network_pruning=0,
                itersize=1000,
                limit=None,
                mode='networkx',
                overwrite=False,
                mutual_overwrite=False,
            )

        # Cluster Directed Retweet Graph
        if False:
            # matrix_npz = '../Datasets/Comparison_Graph/directed_retweet_matrix_qanon.npz'
            # input_gexf = '/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Election_Graph/Direct_Graphs/Election_Network_all_direct_99_retweet.gexf'
            input_gexf = '../Datasets/Comparison_Graph/Election_Network_whole_election_Direct_Retweet_2.gexf'
            output_partitions = '../Datasets/Comparison_Graph/directed_retweet_2_whole_election_partition.pkl'
            cluster_directed_retweet_gexf(input_gexf, output_partitions)

        if False:
            input_csv = '../Datasets/Comparison_Graph/whole_election_relations.csv'
            user_pkl = '/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Intermediate_Files/whole_election_user_connects_restricted_terms.pkl'
            output_csv = '../Datasets/Comparison_Graph/whole_election_node_data.csv'
            get_indegree_list_csv(input_csv, user_pkl, output_csv)

        if False:
            input_filepath = '../Datasets/Comparison_Graph/infomap_clustering/whole_election_relations.tree'
            output_csv = '../Datasets/Comparison_Graph/infomap_clusters_less.csv'
            load_infomap_clusters(input_filepath, output_csv)

        # Find biggest node in each cluster.
        if False:
            # input_gexf = '/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Election_Graph/Direct_Graphs/Election_Network_all_direct_99_retweet.gexf'
            # input_gexf = '../Datasets/Comparison_Graph/Election_Network_whole_election_Direct_Retweet_2.gexf'            
            # output_partitions = '../Datasets/Comparison_Graph/directed_retweet_2_whole_election_partition.pkl'
            # output_degrees = '../Datasets/Comparison_Graph/direct_2_degree_cluster.csv'

            cluster_csv = '../Datasets/Comparison_Graph/infomap_clusters_less.csv'
            indegree_csv = '../Datasets/Comparison_Graph/whole_election_node_data.csv'
            output_degrees = '../Datasets/Comparison_Graph/direct_1_degree_cluster.csv'
            get_highest_degree_clusters(cluster_csv, indegree_csv, output_degrees, top=10)

        if False:
            cluster_csv = '../Datasets/Comparison_Graph/infomap_clusters_less.csv'
            network_csv = '../Datasets/Comparison_Graph/whole_election_relations.csv'
            output_graph = '../Datasets/Comparison_Graph/Cluster_17_Direct_1_Graph.gexf'
            isolate_cluster(cluster_csv, network_csv, output_graph, cluster=17)

        if False:
            case_study = 'Case 1'
            case_study_filepath = '../Datasets/Case_Study_Graphs/Case_1_Layout.gexf'
            download_case_study(case_study, case_study_filepath)

        if False:
            cluster_csv = '../Datasets/Comparison_Graph/infomap_clusters_less.csv'
            indegree_csv = '../Datasets/Comparison_Graph/whole_election_node_data.csv'
            output_degrees = '../Datasets/Comparison_Graph/direct_1_degree_top1000.csv'
            get_highest_degree_nodes(cluster_csv, indegree_csv, output_degrees, top=1000)

        if False:
            cluster_csv = '../Datasets/Comparison_Graph/infomap_clusters_less.csv'
            indegree_csv = '../Datasets/Comparison_Graph/whole_election_node_data.csv'
            cases = ['/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Parameter_Space/Election_Network_whole_election_mutual_10000_1_retweet.gexf']
            cases += ['/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Case_Study_Graphs/Case_1_n100_s5.gexf']
            cases += ['/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Case_Study_Graphs/Case_3_n25_s25.gexf']
            for case in cases:
                get_percent_of_top_1000(indegree_csv, case)

        if False:
            cluster_csv = '../Datasets/Comparison_Graph/infomap_clusters_less.csv'
            indegree_csv = '../Datasets/Comparison_Graph/whole_election_node_data.csv'
            output_csv = '../Datasets/Comparison_Graph/infomap_cluster_distributions_case_'
            cases = ['/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Case_Study_Graphs/Case_1_n100_s5_layout.gexf']
            cases += ['/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Case_Study_Graphs/Case_2_n10000_s1_layout.gexf']
            cases += ['/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Case_Study_Graphs/Case_3_n25_s25_layout.gexf']
            for idx, case in enumerate(cases):
                get_cluster_distribution_infomap(indegree_csv, cluster_csv, case, f'{output_csv}{idx + 1}.csv')

        if False:
            cluster_csv = '../Datasets/Comparison_Graph/infomap_clusters_less.csv'
            indegree_csv = '../Datasets/Comparison_Graph/whole_election_node_data.csv'
            # case_study_filepath = '../Datasets/Case_Study_Graphs/Case_1_Layout.gexf'
            # case_study_filepath = '/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Parameter_Space/Election_Network_whole_election_mutual_10000_1_retweet.gexf'
            case_study_filepath = '/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Case_Study_Graphs/Case_3_n25_s25.gexf'
            # case_study_filepath = '/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Case_Study_Graphs/Case_1_n100_s5.gexf'
            get_cluster_percentage_of_total(indegree_csv, cluster_csv, case_study_filepath)

        if False:
            user_pkl = '/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Intermediate_Files/whole_election_user_connects_restricted_terms.pkl'
            output_degrees = '../Datasets/Comparison_Graph/direct_2_degree_cluster.csv'
            output_csv = '../Datasets/Comparison_Graph/direct_2_degree_cluster_labeled.csv'
            relabel_highest_clusters(output_degrees, user_pkl, output_csv)

        if True:
            user_pkl = '/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/Final_Draft_Graph/Intermediate_Files/whole_election_user_connects_restricted_terms.pkl'
            input_graph = '../Datasets/Comparison_Graph/Cluster_17_Direct_1_Graph.gexf'
            output_graph = '../Datasets/Comparison_Graph/Cluster_17_Direct_1_Graph_labelled.gexf'
            relabel_graph(input_graph, output_graph, user_pkl)

        if False:
            clusters = []

        if False:
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            matrix_npz = '../Datasets/Comparison_Graph/directed_retweet_matrix.npz'
            nodelist = '../Datasets/Comparison_Graph/directed_retweet_nodelist.txt'
            create_directed_retweet_gexf(dict_pkl, matrix_npz, nodelist)

        if False:
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            # dict_pkl = '/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/QAnon_Graph/qanon_01-01-2020_02-01-2020_dict_connects.pkl'
            output_csv = '../Datasets/Comparison_Graph/whole_election_relations.csv'
            convert_pkl_to_csv(dict_pkl, output_csv)

        if False:
            output_csv = '../Datasets/Comparison_Graph/whole_election_relations.csv'
            matrix_npz = '../Datasets/Comparison_Graph/directed_retweet_matrix.npz'
            nodelist = '../Datasets/Comparison_Graph/directed_retweet_nodelist.txt'
            create_directed_retweet_gexf_2(output_csv, matrix_npz, nodelist)

        if False:
            matrix_npz = '../Datasets/Comparison_Graph/directed_retweet_matrix_qanon.npz'
            user_pkl = '/mnt/nfs/jupiter/shared/jupyterhub/public/shared_code/RepeatOffenders_Research/CosharingGraph_Results/Datasets/QAnon_Graph/qanon_01-01-2020_02-01-2020_user_connects.pkl'            
            get_highest_indegree_nodes(matrix_npz, user_pkl, output_nodes)

    exit()

    # Methods / Dataset

    if True:

        """ The time estimates on progress bars for these functions can be deceiving, as they significantly
            speed up over time. That said, on our not-insignificant computing structure, the runtime of the
            longer functions ranges from several hours to a few days.
        """

        """ Create edge-and-link data files streaming from a large CSV file of all retweets in this dataset.
            The input and output files are very large, and this process can take a while -- you may want to
            download the results of this step in the following code block. The tar.gz file is ~20GB, and
            the unzipped version is ~60GB. The link data files downloaded later are significantly smaller
            and we recommend using them.
        """
        if False:
            output_filepath = '../Datasets/Final_Paper_Statistics_Figures/Open_Source_Data.tar.gz'
            download_raw_tweets(output_filepath)

        """ Stream link and label dictionaries from raw csv.
        """
        if False:
            raw_data = '../Datasets/Final_Paper_Statistics_Figures/Open_Source_Data.csv'
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            user_pkl = '../Datasets/Intermediate_Files/whole_election_user_connects_restricted_terms_anon.pkl'
            load_csv_connection_data(raw_data, dict_pkl, user_pkl)

        """ Or, download link data files remotely.
        """
        if True:
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            user_pkl = '../Datasets/Intermediate_Files/whole_election_user_connects_restricted_terms_anon.pkl'
            download_dataset(dict_pkl, user_pkl, overwrite=False)

        # Dataset tweet and user totals.
        if True:
            user_pkl = '../Datasets/Intermediate_Files/whole_election_user_connects_restricted_terms_anon.pkl'
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            dataset_stats_csv = '../Datasets/Final_Paper_Statistics_Figures/dataset_stats_overall.csv'
            dataset_statistics(user_pkl, dict_pkl, dataset_stats_csv)

        """ Generate Shared Engagement Graphs across different s values, n=3. Higher values of n can be generated
            from cached files created via this process. Note that his process can take a long amount of time depending
            on the s values simulated and the size of the dataset.
        """
        if True:
            user_pkl = '../Datasets/Intermediate_Files/whole_election_user_connects_restricted_terms_anon.pkl'
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            output_dir = '../Datasets/Intermediate_Files'
            network_dir = '../Datasets/Parameter_Space'
            prefix = 'whole_election'
            s_range = [5, 500]  # This will take a long time. You can try [20, 30] for something more manageable.
            generate_shared_engagement_graphs(
                output_dir, network_dir, user_pkl, dict_pkl,
                prefix='whole_election', mutual_range=s_range, connection_range=[3, 4], overwrite=False,
            )

    # Introduction

    if True:

        # Generate Direct Graph for Comparison [Figure 1]
        user_pkl = '../Datasets/Intermediate_Files/whole_election_user_connects_restricted_terms_anon.pkl'
        dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
        output_filepath = '../Datasets/Comparison_Graph/Election_Network_whole_election_Direct_Retweet_50.gexf'
        generate_network_gexf(
            database_name=None,
            db_config_file=None,
            input_json_dir=None,
            output_network_file=output_filepath,
            save_pkl=True,
            load_from_pkl=True,
            load_from_gexf=False,
            input_network_file=None,
            dict_pkl_file=dict_pkl,
            users_pkl_file=user_pkl,
            mutual_pkl_file=None,
            table_name='tweets',
            connection_type='retweet',
            link_type='direct',
            conditions=[],
            attributes=None,
            label='screen_name',
            connection_limit=50,
            mutual_limit=0,
            network_pruning=0,
            itersize=1000,
            limit=None,
            mode='networkx',
            overwrite=False,
            mutual_overwrite=False,
        )

    # Case 1: s=1, n=10000
    if True:

        """ Create s=1 Graph, n=10000. Earlier, we create graphs by caching temporary files,
            but for s=1, this temporary cached file would be too large. We run the engagement
            graph creation function with a slightly different parameter set to optimize memory
            usage for this low value of s [Figure 3].
        """
        if True:
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            user_pkl = '../Datasets/Intermediate_Files/whole_election_user_connects_restricted_terms_anon.pkl'
            output_dir = '../Datasets/Intermediate_Files'
            network_dir = '../Datasets/Parameter_Space'
            prefix = 'whole_election'
            s_1_graph(dict_pkl, user_pkl, output_dir, network_dir, prefix, connection_limit=10000)

        """ The previous command will create an un-laid-out and unclassified version of the network graph we use
            in this paper. Layout and classification are done manually in Gephi, so we download that version of
            the graph remotely here, and proceed. You can layout and cluster your own verson of the previous graph
            and proceed accordingly, although you may have to change the identifiers for some clusters.
        """
        if True:
            case_study = 'Case 1'
            case_study_filepath = '../Datasets/Case_Study_Graphs/Case_1_Layout.gexf'
            download_case_study(case_study, case_study_filepath)

        # Count the number of nodes distributed across clusters.
        if True:
            base_graph = '../Datasets/Case_Study_Graphs/Case_1_Layout.gexf'
            output_csv = '../Datasets/Final_Paper_Statistics_Figures/Case_1_Cluster_Totals.csv'
            count_nodes_in_clusters(base_graph, output_csv)

        # Get statistics related to cluster crossing between nodes.
        if True:
            base_graph = '../Datasets/Case_Study_Graphs/Case_1_Layout.gexf'
            output_directory = '../Datasets/Final_Paper_Statistics_Figures'
            output_csv = output_directory + '/CrossCluster_Measurements_Election_10000_1.csv'
            output_bridge_csv = output_directory + '/CrossCluster_Measurements_Election_Bridge_10000_1.csv'
            output_cluster_csv = output_directory + '/CrossCluster_Measurements_Election_Clusters_10000_1.csv'
            output_cumulative_csv = output_directory + '/CrossCluster_Measurements_Election_Cumulative_10000_1.csv'
            cross_cluster_measurement(base_graph, output_csv, output_bridge_csv, output_cluster_csv, output_cumulative_csv, only_clusters=[1, 2])

        # Get accounts linked to "bridging" nodes.
        if True:
            output_directory = '../Datasets/Final_Paper_Statistics_Figures'
            input_csv = output_directory + '/CrossCluster_Measurements_Election_Cumulative_10000_1.csv'
            input_network = '../Datasets/Case_Study_Graphs/Case_1_Layout.gexf'
            output_csv = output_directory + '/Bridge_Links_10000_1.csv'
            get_linked_users(input_csv, input_network, output_csv)

        """ Invert dict_pkl, creating a dictionary linking in-network users to the list
            of users that retweeted them. This function make take a long time.
        """
        if False:
            output_directory = '../Datasets/Final_Paper_Statistics_Figures'
            input_csv = output_directory + '/Bridge_Links_10000_1.csv'
            input_graph = '../Datasets/Case_Study_Graphs/Case_1_Layout.gexf'
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            output_pkl = output_directory + '/Link_User_10000_1_Data.pkl'
            generate_linked_user_data(input_csv, dict_pkl, output_pkl, input_graph)

        """ Get all retweets and post times from input CSV file. This requires downloading the full raw tweet
            data from the Dataset code block. It will also take a long time, the following block can let you
            directly download the output of this function.
        """
        # Get Linking Tweets Origin
        if False:
            raw_csv = '../Datasets/Final_Paper_Statistics_Figures/Open_Source_Data.csv'
            linking_csv = '../Datasets/Final_Paper_Statistics_Figures/Bridge_Links_10000_1.csv'
            tweet_filepath = '../Datasets/Final_Paper_Statistics_Figures/Bridge_Tweet_Data_Anon.csv'
            extract_linked_tweets(raw_csv, linking_csv, tweet_filepath)

        """ The above two functions can take a while, so we also provide you a data file with tweet IDs, user IDs, and
            tweet creation dates for download. This file is produced by the previous function.
        """
        if True:
            tweet_filepath = '../Datasets/Final_Paper_Statistics_Figures/Bridge_Tweet_Data_Anon.csv'
            bridge_connect_data_filepath = '../Datasets/Final_Paper_Statistics_Figures/Link_User_10000_1_Data.pkl'
            download_bridge_user_data(tweet_filepath, bridge_connect_data_filepath)

        # Create left/right distributions for tweets on each of the
        if True:
            output_directory = '../Datasets/Final_Paper_Statistics_Figures'
            tweet_csv = '../Datasets/Final_Paper_Statistics_Figures/Bridge_Tweet_Data_Anon.csv'
            input_graph = '../Datasets/Case_Study_Graphs/Case_1_Layout.gexf'
            input_csv = output_directory + '/Bridge_Links_10000_1.csv'
            output_csv = output_directory + '/Link_User_Tweets_Preprocessed.csv'
            input_pkl = output_directory + '/Link_User_10000_1_Data.pkl'
            combine_reduce_tweets(tweet_csv, input_pkl, input_csv, output_csv, input_graph)

        # Graph Shared User Data [Figure 4]
        if True:
            output_directory = '../Datasets/Final_Paper_Statistics_Figures'
            input_csv = output_directory + '/Link_User_Tweets_Preprocessed.csv'
            plot_prefix = output_directory + '/Link_User_Plot'
            graph_tweet_allignment(input_csv, plot_prefix)

    # Case 2
    if True:

        # Create s=5, n=100 Graph [Figure 5]
        if True:
            input_graph = '../Datasets/Parameter_Space/Election_Network_whole_election_mutual_3_5_retweet.gexf'
            threshold = 100
            output_graph = '../Datasets/Case_Study_Graphs/Case_2_n100_s5.gexf'
            n_filter_graph(input_graph, threshold, output_graph)

        """ The previous command will create an un-laid-out and unclassified version of the network graph we use
            in this paper. Layout and classification are done manually in Gephi, so we download that version of
            the graph remotely here, and proceed. You can layout and cluster your own verson of the previous graph
            and proceed accordingly, although you may have to change the identifiers for some clusters.
        """
        if True:
            case_study = 'Case 2'
            case_study_filepath = '../Datasets/Case_Study_Graphs/Case_2_Layout.gexf'
            download_case_study(case_study, case_study_filepath)

        # Calculate intersections between all two-class pairs in Case 2. [Table 1]
        if True:
            for c1, c2 in combinations([0, 1, 4], 2):
                base_graph = '../Datasets/Case_Study_Graphs/Case_2_Layout.gexf'
                output_directory = '../Datasets/Final_Paper_Statistics_Figures'
                output_csv = output_directory + '/CrossCluster_Measurements_Election_100_5.csv'
                output_bridge_csv = output_directory + '/CrossCluster_Measurements_Election_Bridge_100_5.csv'
                output_cluster_csv = output_directory + '/CrossCluster_Measurements_Election_Clusters_100_5.csv'
                output_cumulative_csv = output_directory + f'/CrossCluster_Measurements_Election_Cumulative_100_5_{c1}_{c2}.csv'
                cross_cluster_measurement(base_graph, output_csv, output_bridge_csv, output_cluster_csv, output_cumulative_csv, only_clusters=[c1, c2])

    # Case 3:
    if True:

        # Create s=25, n=25 Graph [Figure 6]
        if True:
            input_graph = '../Datasets/Parameter_Space/Election_Network_whole_election_mutual_3_25_retweet.gexf'
            threshold = 25
            output_graph = '../Datasets/Case_Study_Graphs/Case_3_n25_s25.gexf'
            n_filter_graph(input_graph, threshold, output_graph)

        """ The previous command will create an un-laid-out and unclassified version of the network graph we use
            in this paper. Layout and classification are done manually in Gephi, so we download that version of
            the graph remotely here, and proceed. You can layout and cluster your own verson of the previous graph
            and proceed accordingly, although you may have to change the identifiers for some clusters.
        """
        if True:
            case_study = 'Case 3'
            case_study_filepath = '../Datasets/Case_Study_Graphs/Case_3_Layout.gexf'
            download_case_study(case_study, case_study_filepath)

        """ Due to data-sharing restrictions, we cannot share follower/following data for each our nodes directly.
            We instead download the follower/following data directly. Follower/following counts are taken from the
            last time a user posted a tweet in our database.
        """
        if True:
            output_filepath = '../Datasets/Final_Paper_Statistics_Figures/User_Follower_Data.csv'
            download_follower_data(output_filepath)

        # Merge follower info with user data. [Figure 7]. Figure consists of scatter plots generated in Tableau.
        if True:
            output_directory = '../Datasets/Final_Paper_Statistics_Figures'
            reference_graph = '../Datasets/Case_Study_Graphs/Case_3_Layout.gexf'
            input_csv = output_directory + '/User_Follower_Data.csv'
            output_csv = output_directory + '/User_Follower_Data_Extended.csv'

            extend_follower_data(reference_graph, input_csv, output_csv)

        # Get intersection data for all two-cluster combinations in Case 3.
        if True:
            for c1, c2 in combinations([0, 1, 3, 4, 6, 7], 2):
                base_graph = '../Datasets/Case_Study_Graphs/Case_3_Layout.gexf'
                output_directory = '../Datasets/Final_Paper_Statistics_Figures'
                output_csv = output_directory + '/CrossCluster_Measurements_Election_25_25.csv'
                output_bridge_csv = output_directory + '/CrossCluster_Measurements_Election_Bridge_25_25.csv'
                output_cluster_csv = output_directory + '/CrossCluster_Measurements_Election_Clusters_25_25.csv'
                output_cumulative_csv = output_directory + f'/CrossCluster_Measurements_Election_Cumulative_25_25_{c1}_{c2}.csv'
                cross_cluster_measurement(base_graph, output_csv, output_bridge_csv, output_cluster_csv, output_cumulative_csv, only_clusters=[c1, c2])

        # Characterize the level of self-linking in all clusters.
        if True:
            input_graph = '../Datasets/Case_Study_Graphs/Case_3_Layout.gexf'
            dict_pkl = '../Datasets/Intermediate_Files/whole_election_dict_connects_restricted_terms.pkl'
            output_csv = '../Datasets/Final_Paper_Statistics_Figures/Self_Link_Data.csv'
            characterize_self_links(dict_pkl, input_graph, [0, 1, 3, 4, 6, 7], output_csv, connection_limit=25)

    # Synthesis
    if True:

        """ Download landmarks for labeling clusters in subsequent code block.
        """
        if True:
            output_filepath = '../Datasets/Final_Paper_Statistics_Figures/cluster_reference.txt'
            download_clustering_reference(output_filepath)

        """ Create data for cluster boundary map. [Figure 8]. This function will take a long time to run, and presumes
            the prior function, generate_shared_engagement_graphs, has already been run.
        """
        if False:
            input_dir = '../Datasets/Parameter_Space/'
            landmarks = '../Datasets/Final_Paper_Statistics_Figures/cluster_reference.txt'
            output_volume_stats = '../Datasets/Final_Paper_Statistics_Figures/Cluster_Existence_Counts.csv'
            cluster_graphs_volume_threshold(input_dir, landmarks, output_volume_stats, bootstrap_iterations=3)

        """ Download results of the previous function, which can take a very long time to run depending on inputs.
        """
        if True:
            output_volume_stats = '../Datasets/Final_Paper_Statistics_Figures/Cluster_Existence_Counts.csv'
            download_cluster_boundaries(output_volume_stats)

        """ Create cluster boundary map. [Figure 9].
        """
        if True:
            output_volume_stats = '../Datasets/Final_Paper_Statistics_Figures/Cluster_Existence_Counts.csv'
            output_viz_filepath = '../Datasets/Final_Paper_Statistics_Figures/Cluster_Boundary_Viz.csv'
            visualize_cluster_boundaries(output_volume_stats, output_viz_filepath)

        """ We provide suspension data generated on 9/23/21 here.
        """
        if True:
            output_filepath = '../Datasets/Final_Paper_Statistics_Figures/Suspended_Users_092321.csv'
            download_suspensions(output_filepath)

        case_studies = {'Case 1': '../Datasets/Case_Study_Graphs/Case_1_Layout.gexf',
                        'Case 2': '../Datasets/Case_Study_Graphs/Case_2_Layout.gexf',
                        'Case 3': '../Datasets/Case_Study_Graphs/Case_3_Layout.gexf'}

        """ We calculate suspension statistics and create .gexf files with nodes colored according to whether
            they have been suspended. [Figure 9]
        """
        if True:
            output_dir = '../Datasets/Suspension_Graphs/'
            suspended_users = '../Datasets/Final_Paper_Statistics_Figures/Suspended_Users_092321.csv'
            suspension_info = '../Datasets/Final_Paper_Statistics_Figures/Suspension_Statistics.csv'
            suspension_statistics(case_studies, output_dir, suspended_users, suspension_info)
