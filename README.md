# Shared Engagement Projections

![A sample shared engagement projection](./README_Image.png?raw=true "Shared Engagement Projections")

A data visualization Docker container for shared engagement projection networks. Created by Joseph S. Schafer and Andrew Beers of 
the University of Washington Center for an Informed Public.
##Table of Contents
1. [About](#about)
2. [Installation](#installation)
3. [Full Documentation](#full-documentation)
   1. [INPUT OPTIONS](#input-options)
   2. [OUTPUT OPTIONS](#output-options)
   3. [OTHER OPTIONS](#other-options)
4. [Example Usage](#example-usage)
5. [Contact](#contact)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)

## About

This Github repository contains code for creating shared engagement projection graphs and visualizations, per the paper "Shared Engagement Projections for the 2020 United States Election: A Network Visualization Case Study and Toolkit." A shared engagement projection is a transformation of a directed network of social media engagements on entitites into an undirected projection of shared engagements between entities. For example, one could take a dataset of accounts retweeting other accounts on Twitter, and create a shared engagement projection where links between accounts denote that they are retweeted by the same intermediate accounts. Intuitively, edges in these projections maps shared audiences of engaged users. Creating these networks can reduce the size and complexity of social network visualizations on extremely large engagement datasets. Please read our paper (currently under review) for additional details and examples.

## Installation
This software uses Docker to run its code. Docker is a service that allows you to run pre-packaged code in "Docker containers" without having to install additional dependencies outside of the Docker softwar itself. Here is how you can get set up with Docker.

1. Install Docker from Docker's website here: https://www.docker.com/get-started. Follow instructions on that link to get Docker set up properly on your workstation.
2. Use the command `docker pull ghcr.io/uwcip-research/shared-engagement-projection:latest` to pull the SharedEngagement Docker container.
3. Follow the documentation below to run your Docker container on directed network datasets in CSV and GEXF formats.

## Full Documentation
To run this Docker container, enter the following command into the command line of the machine where you pulled your Docker container. Replace the sections in brackets with parameters as described in the documentation below.

```shell
docker run --rm -v [INPUT_DIRECTORY]:/input_data [OUTPUT_DIRECTORY]:/output_data shared-engagement-projection [INPUT OPTIONS] [OUTPUT OPTIONS] [OTHER OPTIONS]

```

* ``-v [INPUT_DIRECTORY]:/input_data``
To use Docker, you have to map data on your internal machine to a location within the docker container. This happens with the "-v" parameter. On the left side of the "-v" parameter is a directory on your local machine. On the right side of this parameter is a directory on the inside of the Docker container. Change the left side of this command to the full path of the directory that contains that data you want to visualize. For example, if your data is in "/home/andrew/data/graph/example.csv,' then you will want to put the directory '/home/andrew/data/graph' on the left side of "-v," where [INPUT_DIRECTORY] is. Similarly, if you are storing your outputs in `/home/andrew/outputs`, you will want to put the directory '/home/andrew/outputs' on the left side of "-v," where [OUTPUT_DIRECTORY] is. While you can change the right side of "-v," i.e. the internal Docker directory, we suggest you leave the right side of the values as the default values, "/input_data" and "/output_data".

### INPUT OPTIONS
In the INPUT_OPTIONS section, you must specify one of two sets of parameters. The first parameter is for if you are inputting data using a CSV file. This CSV file should be in the following format.

| source | target |
|--------|--------|
| 1      | 2      |
| 3      | 1      |

SOURCE and TARGET should correspond to unique identifiers for each node in your dataset. Source nodes direct to target nodes. SOURCE_NAME and TARGET_NAME are optional aliases for SOURCE and TARGET, which if present, will be used as labels in the resulting visualization. After you have specified your CSV file, use the following parameters:

``--input-data [filename] --n [val1] --s [val2]``

--input-data should link to the CSV file you would like to be processed. Note here that because of the way Docker processes files, this filename must be relative to the directory inside the Docker container, which unless you have changed it in the previous parameters, should be "/input_data." So, using the previous example, if your input file is "/home/andrew/data/graph/example.csv" and you mapped the directory "/home/andrew/data/graph," then your --input-data filepath here should be "/input_data/example.csv."

The second two parameters, --n and --s, correspond to parameters that define the type of shared engagement projection that you want to make. Intuitively, you can understand these parameters as such: in the projection, nodes A and B are linked if at least *n* other nodes engage with both of them at least *s* times. An extended discussion of how these two parameters affect output visualizations can be found in our paper. We suggest a starting value of n=2, s=2, which you can modify later to generate different visualizations. If your dataset is very large (>100M nodes), you may want to raise these parameter values to constrict the size of the network.

### OUTPUT OPTIONS
This code can generate several different forms of output. You can specify as many as you would like with the following parameters.

* ``--output-graph [filename]`` Outputs a .gexf file to filename, usable in Gephi and other network visualization software packages. This .gexf file contains the data of the projected graph.
* ``--output-pdf [filename]`` Outputs a .pdf file to filename. This PDF file displays the full visualized graph as an image.
* ``--output-csv [base-filename.csv]`` Outputs two .csv files, one to [base-filename]_nodes.csv and one to [base-filename]_edges.csv, corresponding to the node and edge tables for the projection graph.
* ``--output-all [base-filename]`` **NOTE:** Not compatible with other outputs. This is a shortcut which outputs all supported output formats. Therefore, this is equivalent to writing ``--output-graph [base-filename.gexf] --output-pdf [base-filename.pdf] --output-csv [base-filename.csv]``. 

Note that as with the inputs, your output filepaths should be relative to the directory mapped inside the Docker container, which if left as default, should be "/input_data." So, if your input file is "/home/andrew/data/graph/example.csv," and you want to create a visualization at "/home/andrew/data/graph/visualization/output_viz.pdf," then the filename that you enter above should be "/input_data/visualization/output_viz.pdf."

### OTHER OPTIONS
``--err [filename]``: If your input data causes an error, you can view the text of this error optionally at this filepath.

``--data-dir [directoryname]``: If this is specified, cached data files for unique parameter sets will be created and saved in this directory. This is highly recommended for faster performance *if* several runs are to be conducted using the same value of *s*, but different values of *n*.

``--python-only``: Specifying this parameter will not run the visualization portion of the code, which is run via Gephi and Java. You can use this to still get output from the code, if the visualization portion of the code is encountering errors.

``--min-size [val]`` and ``--max-size [val].`` This sets the minimum and maximum size of nodes visualized in Gephi. The default values are 1 and 100, respectively.

## Example Usage

You can test out this package with a sample Reddit dataset we have downloaded from [Stanford Network Analysis Project](https://snap.stanford.edu/data/soc-RedditHyperlinks.html). This dataset contains information about when one subreddit's posts contained links to content from another subreddit.

1. Download the [soc-redditHyperlinks-body.tsv](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) file from this link.
2. Using a spreadsheet editor of your choice, rename the source and target columns to 'source' and 'target' respectively, and resave as a CSV file. In our case, you're going to rename "SOURCE_SUBREDDIT" and "TARGET_SUBREDDIT" to "source" and "target" respectively. You can skip this step by download the modified data [here](https://drive.google.com/file/d/1x-G1ZM1LVGWwA3yznkgSxRbLhCaIbQfO/view?usp=sharing).
3. Identify the path to the folder that contains your data. This could be `/home/andrew/data/example_data`. We'll assume it is for the rest of this example.
4. Identify the parameters you want to use. Recall that the parameters N and S can be interrupted as follows: "Each edge in the projected graph represents two nodes that both are connected to by N other nodes at least S times each." In this case, we'll pick N=2 and S=2, so, for our data, "Two subreddits are connected if at least two other subreddits linked out to both of them at least two times."
5. Identify the path to where you want to output your results. For this case, we'll assume that's `/home/andrew/example_output`.
6. Following the instructions from the documentation, format your Docker commands as follows:

``docker run --rm -v /home/andrew/data/example_data:/input_data /home/andrew/example_output:/output_data -it shared-engagement-projection --input-data /input_data/soc-redditHyperlinks-body.csv --output-all /output_data/example_case -n 2 -s 2``

This would create 4 output files, a GEXF of the network, a PDF containing the network visualization, and two CSVs which together contain the network structure. While there is some randomness to the output, the PDF outputs should similar to this:

![A sample shared engagement projection of Reddit data.](./tutorial_example.jpg?raw=true "Shared Engagement Projections")

## Contact
For questions about the code, please contact Joseph S. Schafer at [schaferj@uw.edu](mailto:schaferj@uw.edu), or on Twitter [@joey__schafer](https://twitter.com/Joey__Schafer)
## Citation
The publication associated with this code is under review. Please contact albeers@uw.edu or schaferj@uw.edu if you would like to reference this work beforehand.
## Acknowledgements
This work is supported by the Center for an Informed Public at the University of Washington, the John S and James L Knight Foundation, Craig Newmark Philanthropies, and the Omidyar Network. We would also like to acknowledge Paul Lockaby of the University of Washington and Clément Levallois from EM Lyon Business School for their valuable advice in completing this project.
