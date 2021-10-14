import org.gephi.appearance.api.AppearanceController;
import org.gephi.appearance.api.AppearanceModel;
import org.gephi.appearance.api.Function;
import org.gephi.appearance.api.Partition;
import org.gephi.appearance.api.PartitionFunction;
import org.gephi.appearance.plugin.PartitionElementColorTransformer;
import org.gephi.appearance.plugin.RankingNodeSizeTransformer;
import org.gephi.appearance.plugin.palette.Palette;
import org.gephi.appearance.plugin.palette.PaletteManager;
import org.gephi.graph.api.Column;
import org.gephi.graph.api.Graph;
import org.gephi.graph.api.GraphController;
import org.gephi.graph.api.GraphModel;
import org.gephi.graph.api.Node;
import org.gephi.io.exporter.api.ExportController;
import org.gephi.io.exporter.plugin.ExporterSpreadsheet;
import org.gephi.io.exporter.spi.GraphExporter;
import org.gephi.io.exporter.spi.GraphFileExporterBuilder;
import org.gephi.io.importer.api.Container;
import org.gephi.io.importer.api.EdgeDirectionDefault;
import org.gephi.io.importer.api.ImportController;
import org.gephi.io.processor.plugin.DefaultProcessor;
import org.gephi.layout.plugin.forceAtlas2.ForceAtlas2;
import org.gephi.preview.api.PreviewController;
import org.gephi.preview.api.PreviewModel;
import org.gephi.preview.api.PreviewProperty;
import org.gephi.preview.types.EdgeColor;
import org.gephi.project.api.ProjectController;
import org.gephi.project.api.Workspace;
import org.gephi.statistics.plugin.ConnectedComponents;
import org.gephi.statistics.plugin.Degree;
import org.gephi.statistics.plugin.Modularity;
import org.gephi.statistics.plugin.builder.ConnectedComponentsBuilder;
import org.gephi.statistics.plugin.builder.ModularityBuilder;
import org.gephi.statistics.spi.Statistics;
import org.openide.util.Lookup;

import java.awt.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class GephiInterface {
  public static void main(String[] args) throws FileNotFoundException {

    // Parse command line arguments passed to Java
    CliParams params = new CliParams(args);

    // Make graph based off of arguments
    makeFile(params);
  }

  // Primary method for making the visualized graph.
  // Takes a CliParams object, consisting of the values parsed from the command-line in an
  // easy-to-access format
  public static void makeFile(CliParams params) {
    //Get local copies
    Map<String, String> nameToClass = params.getNameToClass();
    Map<String, Integer> classToInteger = params.getClassToInteger();
    Map<Integer, String> integerToClass = params.getIntegerToClass();
    Map<String, Color> classToColor = params.getClassToColor();

    //Initialization
    ProjectController pc = Lookup.getDefault().lookup(ProjectController.class);
    pc.newProject();
    Workspace workspace = pc.getCurrentWorkspace();
    AppearanceController appearanceController =
        Lookup.getDefault().lookup(AppearanceController.class);
    AppearanceModel appearanceModel = appearanceController.getModel();


    // Import Graph data from passed/temporary GEXF
    ImportController importController = Lookup.getDefault().lookup(ImportController.class);
    Container container;
    try {
      container = importController.importFile(params.getInputFile());
      container.getLoader().setEdgeDefault(EdgeDirectionDefault.UNDIRECTED);
      container.getLoader().setAllowAutoNode(false);
    } catch (Exception ex) {
      ex.printStackTrace();
      return;
    }
    importController.process(container, new DefaultProcessor(), workspace);
    GraphModel graphModel =
        Lookup.getDefault().lookup(GraphController.class).getGraphModel();
    Graph graph = graphModel.getGraph();

    // Print size of graph
    System.out.println("Nodes: " + graph.getNodeCount());
    System.out.println("Edges: " + graph.getEdgeCount());

    // Modularity class coloring
    Column modColumn = getModularityColumn(graphModel);

    // Order of list corresponds to number of mod class, NOT size of classes
    // i.e. even if class 0 is small and class 5 is biggest % of nodes, the color for class
    // 0 will still be listed first

    Function func = appearanceModel.getNodeFunction(graph, modColumn,
        PartitionElementColorTransformer.class);
    Partition p = ((PartitionFunction) func).getPartition();

    Column labelCol = graphModel.getNodeTable().getColumn("label");

    boolean overrideColors = nameToClass != null && nameToClass.size() > 0;
    if (overrideColors) { // Use custom, user-defined color palette (if present)
      Color[] colorArray = new Color[p.size()];
      // Reset all clusters to having light-gray
      Arrays.fill(colorArray, Color.LIGHT_GRAY);

      // Create array counting size of each cluster - this is then sorted to make the order of
      // classes by size be indexed by the number of the class.
      ClassCount[] classCounts = new ClassCount[p.size()];
      for (int i = 0; i < classCounts.length; i++) {
        classCounts[i] = new ClassCount(i);
      }

      // Count incidence of each predefined cluster in each algorithmically determined cluster
      int[][] counts = new int[classToInteger.keySet().size()][p.size()];
      for (Node n : graphModel.getGraph().getNodes()) {
        String label = ((String) n.getAttribute(labelCol)).toLowerCase();
        Integer modClass = (Integer) n.getAttribute(modColumn);
        classCounts[modClass].increment();
        if (nameToClass.containsKey(label)) {
          counts[classToInteger.get(nameToClass.get(label))][modClass] += 1;
        }
      }
      Arrays.sort(classCounts);

      // In reverse order (so that the first class listed has the highest priority by being set
      // last), set the class with the highest # of nodes from each predefined class to the color
      // that class is associated with.
      for (int i = counts.length - 1; i >= 0; i--) {
        int maxIndex = 0;
        for (int j = 1; j < p.size(); j++) {
          if (counts[i][j] > counts[i][maxIndex]) {
            maxIndex = j;
          }
        }
        colorArray[maxIndex] = classToColor.get(integerToClass.get(i));
      }
      Color[] finalColorArray = new Color[p.size()];

      // Set the final colors to correspond correctly
      for (int i = 0; i < classCounts.length; i++) {
        finalColorArray[i] = colorArray[classCounts[i].getModClass()];
      }
      Palette palette = new Palette(finalColorArray);
      p.setColors(palette.getColors());
    } else {
      // Use a default palette for clustering
      PaletteManager paletteManager = PaletteManager.getInstance();

      Palette palette = paletteManager.generatePalette(p.size());
      p.setColors(palette.getColors());

    }
    // Apply
    appearanceController.transform(func);

    // Filter to components containing referenced nodes, if requested
    // Note - nodes in connected, but distinct, clusters will still be present. This only filters
    // Entirely unconnected components.
    if (params.getFilterToRefs() && !(nameToClass == null) && !nameToClass.isEmpty()) {
      Statistics connectedComponents = new ConnectedComponentsBuilder().getStatistics();
      connectedComponents.execute(graphModel);
      Column componentColumn = graphModel.getNodeTable().getColumn(ConnectedComponents.WEAKLY);

      // Set containing the numbers of components that we want to keep
      Set<Integer> componentsToKeep = new HashSet<>();
      // Populate the set
      for(Node n : graph.getNodes()) {
        if (nameToClass.containsKey(((String) n.getAttribute(labelCol)).toLowerCase())) {
          componentsToKeep.add((Integer) n.getAttribute(componentColumn));
        }
      }

      // Remove nodes in separate components
      for(Node n : graph.getNodes().toArray()) {
        if (!componentsToKeep.contains(n.getAttribute(componentColumn))) {
          graph.removeNode(n);
        }
      }
    }

    // Set size on degree if no column specified, otherwise set on that column
    if (params.getColumnToSize().equals("")) {
      Degree degree = new Degree();
      degree.execute(graphModel);
      Column degreeColumn = graphModel.getNodeTable().getColumn(Degree.DEGREE);
      Function sizeFunc = appearanceModel.getNodeFunction(graph, degreeColumn,
          RankingNodeSizeTransformer.class);
      RankingNodeSizeTransformer rankingTransformer = sizeFunc.getTransformer();

      rankingTransformer.setMinSize(params.getSizeMin());
      rankingTransformer.setMaxSize(params.getSizeMax());
      appearanceController.transform(sizeFunc);

    } else {
      Column sizeColumn = graphModel.getNodeTable().getColumn(params.getColumnToSize());
      Function sizeFunc = appearanceModel.getNodeFunction(graph, sizeColumn,
          RankingNodeSizeTransformer.class);
      RankingNodeSizeTransformer rankingTransformer = sizeFunc.getTransformer();
      rankingTransformer.setMinSize(params.getSizeMin());
      rankingTransformer.setMaxSize(params.getSizeMax());
      appearanceController.transform(sizeFunc);
    }


    // Layout using Force Atlas 2, according to specified parameters
    ForceAtlas2 layout = new ForceAtlas2(null);
    layout.setGraphModel(graphModel);
    layout.resetPropertiesValues();
    layout.setEdgeWeightInfluence(params.getForceAtlasParams().getEdgeWeight());
    layout.setGravity(params.getForceAtlasParams().getGravity());
    layout.setStrongGravityMode(params.getForceAtlasParams().getUseStrongGravity());
    layout.setScalingRatio(params.getForceAtlasParams().getScaling());


    layout.initAlgo();
    for (int i = 0; i < 1000 && layout.canAlgo(); i++) {
      layout.goAlgo();
    }
    layout.endAlgo();



    // Set visualization parameters
    PreviewModel previewModel = Lookup.getDefault().lookup(PreviewController.class).getModel();
    previewModel.getProperties().putValue(PreviewProperty.EDGE_THICKNESS, 0.3f);

    previewModel.getProperties().putValue(PreviewProperty.NODE_LABEL_FONT,
        previewModel.getProperties().getFontValue(PreviewProperty.NODE_LABEL_FONT).deriveFont(Font.PLAIN));
    previewModel.getProperties().putValue(PreviewProperty.EDGE_COLOR, new EdgeColor(EdgeColor.Mode.MIXED));
    previewModel.getProperties().putValue(PreviewProperty.NODE_BORDER_WIDTH, 1.0);
    previewModel.getProperties().putValue(PreviewProperty.SHOW_NODE_LABELS, true);
    previewModel.getProperties().putValue(PreviewProperty.EDGE_RESCALE_WEIGHT, true);
    previewModel.getProperties().putValue(PreviewProperty.NODE_LABEL_PROPORTIONAL_SIZE, true);


    // Write to output files
    ExportController ec = Lookup.getDefault().lookup(ExportController.class);
    // PDF for quick visualization
    if (!params.getOutputPdfFileName().equals("")) {
      try {
        ec.exportFile(new File(params.getOutputPdfFileName()));
      } catch (IOException ex) {
        ex.printStackTrace();
        return;
      }
    }
    // GEXF for more in-depth analysis
    if (!params.getOutputGraphFileName().equals("")) {
      try {
        ec.exportFile(new File(params.getOutputGraphFileName()));
      } catch (IOException ex) {
        ex.printStackTrace();
        return;
      }
    }
    // CSVs for pipelining clustering information to other analyses
    if (!params.getOutputCsvBaseFileName().equals("")) {
      ExporterSpreadsheet.ExportTable nodeTable = ExporterSpreadsheet.ExportTable.NODES;

      for (GraphFileExporterBuilder builder :
          Lookup.getDefault().lookupAll(GraphFileExporterBuilder.class)) {
        if (builder.getName().toLowerCase().startsWith("spreadsheet")) {
          GraphExporter exporter = builder.buildExporter();
          ((ExporterSpreadsheet) exporter).setTableToExport(nodeTable);
          exporter.setExportVisible(true);
          try {
            String outFileName = params.getOutputCsvBaseFileName();
            File csvFile = new File(outFileName.substring(0, outFileName.indexOf('.'))+
                "_nodes.csv");
            ec.exportFile(csvFile, exporter);
          } catch (IOException ex) {
            ex.printStackTrace();
            return;
          }
        }
      }

      ExporterSpreadsheet.ExportTable edgeTable = ExporterSpreadsheet.ExportTable.EDGES;
      for (GraphFileExporterBuilder builder :
          Lookup.getDefault().lookupAll(GraphFileExporterBuilder.class)) {
        if (builder.getName().toLowerCase().startsWith("spreadsheet")) {
          GraphExporter exporter = builder.buildExporter();
          ((ExporterSpreadsheet) exporter).setTableToExport(edgeTable);
          exporter.setExportVisible(true);
          try {
            String outFileName = params.getOutputCsvBaseFileName();
            File csvFile = new File(outFileName.substring(0, outFileName.indexOf('.'))+
                "_edges.csv");
            ec.exportFile(csvFile, exporter);
          } catch (IOException ex) {
            ex.printStackTrace();
            return;
          }
        }
      }
    }
  }

  // Helper method to get the modularity class column.
  private static Column getModularityColumn(GraphModel graphModel) {
    Statistics modularity = new ModularityBuilder().getStatistics();
    ((Modularity) modularity).setResolution(1.0);
    modularity.execute(graphModel);
    return graphModel.getNodeTable().getColumn(Modularity.MODULARITY_CLASS);
  }

}
