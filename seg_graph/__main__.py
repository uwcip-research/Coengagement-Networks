import argparse
from . import main

""" Initialization code for the GEXF file generator - parses Python command-line arguments, and then passes this
    to __init__.py
"""
parser = argparse.ArgumentParser(
    description='Create shared engagement projections from CSV data, and visualize the outputs using Gephi in PDF format.',
    usage='''gephi_cluster [<args>] ''')

parser.add_argument('--input-graph', help='GEXF file to input')
parser.add_argument('--input-data', help='CSV file to input')
parser.add_argument('--output-graph', help='GEXF file to output')
parser.add_argument('--output-pdf', help='PDF file to output')
parser.add_argument('--output-csv', help='CSV statistics files to output')
parser.add_argument('--output-all', help='Use all output files with the same base file name')
parser.add_argument('--ref', help='Reference file for cluster labeling')
parser.add_argument('--err', help='Error/runtime events output file')
parser.add_argument('--data-dir', help='Directory for caching intermediate results for faster future runtimes')
parser.add_argument('--trim-unconnected', help='Remove nodes not connected to any canonical clusters', action="store_true")
parser.add_argument('--mutual-limit', '-n', help='n parameter for mutual graphs if using input_data', type=int)
parser.add_argument('--connection-limit', '-s', help='s parameter for mutual graphs if using input_data', type=int)
parser.add_argument('--python-only', '-p', help='do not lay out/cluster the graph, only execute python section', action="store_true")
parser.add_argument('--min-size', help='minimum node size', type=int)
parser.add_argument('--max-size', help='maximum node size', type=int)

args = parser.parse_args()

if args.input_graph is None and args.input_data is None:
    print("Error: No input specified")

if args.input_graph is not None and args.input_data is not None:
    print("Error: Multiple inputs specified")

if args.output_graph is None and args.output_pdf is None and args.output_csv is None and args.output_all is None:
    print("Error: No output specified")

if args.output_all is not None and not (args.output_pdf is None and args.output_graph is None and args.output_csv is None):
    print("Error: Conflicting outputs specified")

if args.ref is None and args.trim_unconnected:
    print("Error: Cannot trim without reference file")

if args.python_only and (args.ref is not None or args.output_pdf is not None or args.output_csv is not None or args.trim_unconnected):
    print("Error: Requesting python-only execution but other arguments require non-python execution")

if args.python_only and args.input_graph is not None:
    print("Error: No operations will be performed (input graph just returned would be expected behavior)")

main(**vars(args))
