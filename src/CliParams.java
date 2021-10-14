import java.awt.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;


// Simple data-holding class which holds the parameters passed to Java in an easy-to-access format
// for subsequent use in GephiInterface/
public class CliParams {
  private String inputFileName; // Input GEXF file  (either directly passed or created by python)
  private String referenceFileName; // reference file for clustering/coloring
  private String outputGraphFileName; // output GEXF file
  private String outputPdfFileName; // output PDF file
  private String outputCsvBaseFileName; // output base name for CSV files
  private boolean filterToRefs; // Boolean for whether to filter out unconnected components
  private String columnToSize; // Which column to use for sizing nodes
  private ForceAtlasParams forceAtlasParams; // Parameters for force atlas layout
  private Map<String, String> nameToClass; // Map going from node name to named class
  private Map<String, Integer> classToInteger; // Map pairing each named class to a unique integer
  private Map<Integer, String> integerToClass; // Inverse of classToInteger
  private Map<String, Color> classToColor; // Map converting each named class to a RGB color
  private File inputFile; // File pointer for the input graph file
  private int sizeMin; // Minimum node size for visualization
  private int sizeMax; // Maximum node size for visualization

  private void initVals() {
    inputFileName = "";
    referenceFileName = "";
    outputGraphFileName = "";
    outputPdfFileName = "";
    outputCsvBaseFileName = "";
    filterToRefs = false;
    columnToSize = "";
    forceAtlasParams = new ForceAtlasParams();
    nameToClass = new HashMap<>();
    classToInteger = new HashMap<>();
    integerToClass = new HashMap<>();
    classToColor = new HashMap<>();
    sizeMin = 1;
    sizeMax = 100;
  }

  public CliParams() {
    initVals();
  }

  public CliParams(String[] args) throws FileNotFoundException {
    initVals();


    //argument parser
    for (int i = 0; i < args.length; i++) {
      if (args[i].equals("--input")) {
        i += 1;
        inputFileName = args[i];
      }
      else if (args[i].equals("--reference")) {
        i+= 1;
        referenceFileName = args[i];
      }
      else if (args[i].equals("--output-graph")) {
        i += 1;
        outputGraphFileName = args[i];
      }
      else if (args[i].equals("--output-pdf")) {
        i += 1;
        outputPdfFileName = args[i];
      }
      else if (args[i].equals("--error")) {
        i += 1;
        System.setErr(new PrintStream(new File(args[i])));
      }
      else if (args[i].equals("--columnsize")) {
        i += 1;
        columnToSize = args[i];
      }
      else if (args[i].equals("--filter")) {
        filterToRefs = true;
      }
      else if (args[i].equals("--output-csv")) {
        i += 1;
        outputCsvBaseFileName = args[i];
      }
      else if (args[i].equals("--size-min")) {
        i += 1;
        sizeMin = Integer.parseInt(args[i]);
      }
      else if (args[i].equals("--size-max")) {
        i += 1;
        sizeMax = Integer.parseInt(args[i]);
      }
      else {
        System.err.println(args[i]);
        System.err.println("Unknown argument flag");
        System.exit(1);
      }
    }

    if (!referenceFileName.equals("")) {
      File refFile = new File(referenceFileName);

      Scanner refScanner = new Scanner(refFile);
      while (refScanner.hasNextLine()) {
        String line = refScanner.nextLine();
        if (line.charAt(0) == '!') {
          //Terms = STRONG, GRAVITY, EDGEWEIGHT, SCALING
          String[] stringInputs = line.substring(1).split("=");
          if (stringInputs.length != 2) {
            System.err.println(line);
            System.exit(1);
          }
          if( stringInputs[0].equalsIgnoreCase("strong")) {
            forceAtlasParams.useStrongGravity = Boolean.parseBoolean(stringInputs[1]);
          } else if (stringInputs[0].equalsIgnoreCase("gravity")) {
            forceAtlasParams.gravity = Double.parseDouble(stringInputs[1]);
          } else if (stringInputs[0].equalsIgnoreCase("edgeweight")) {
            forceAtlasParams.edgeWeight = Double.parseDouble(stringInputs[1]);
          } else if (stringInputs[0].equalsIgnoreCase("scaling")) {
            forceAtlasParams.scaling = Double.parseDouble(stringInputs[1]);
          } else { // Unknown input
            System.err.println(line);
            System.exit(1);
          }

        } else {
          String[] stringInputs = line.split(":");
          if (stringInputs.length != 3) {
            System.err.println(line);
            System.exit(1);
          }
          String[] colorStringArray = stringInputs[1].split(",");

          if (!nameToClass.keySet().contains(stringInputs[0])) {
            nameToClass.put(stringInputs[0], stringInputs[2]);
          }
          if (!classToInteger.keySet().contains(stringInputs[2])) {
            int len = classToInteger.size();
            classToInteger.put(stringInputs[2], len);
            integerToClass.put(len, stringInputs[2]);
            classToColor.put(stringInputs[2], new Color(Integer.parseInt(colorStringArray[0]),
                Integer.parseInt(colorStringArray[1]), Integer.parseInt(colorStringArray[2])));
          }
        }

      }
    }
    this.inputFile = new File(this.getInputFileName());


  }

  public String getInputFileName() {
    return this.inputFileName;
  }

  public String getReferenceFileName() {
    return this.referenceFileName;
  }

  public String getOutputGraphFileName() {
    return this.outputGraphFileName;
  }

  public String getOutputPdfFileName() {
    return this.outputPdfFileName;
  }

  public String getOutputCsvBaseFileName() {
    return this.outputCsvBaseFileName;
  }

  public boolean getFilterToRefs() {
    return this.filterToRefs;
  }

  public String getColumnToSize() {
    return this.columnToSize;
  }

  public ForceAtlasParams getForceAtlasParams() {
    return this.forceAtlasParams;
  }

  //Copy done for security purposes
  public Map<String, String> getNameToClass() {
    Map<String, String> toReturn = new HashMap<>();
    toReturn.putAll(nameToClass);

    return toReturn;
  }

  public Map<String, Integer> getClassToInteger() {
    Map<String, Integer> toReturn = new HashMap<>();
    toReturn.putAll(classToInteger);

    return toReturn;
  }

  public Map<Integer, String> getIntegerToClass() {
    Map<Integer, String> toReturn = new HashMap<>();
    toReturn.putAll(integerToClass);

    return toReturn;
  }

  public Map<String, Color> getClassToColor() {
    Map<String, Color> toReturn = new HashMap<>();
    toReturn.putAll(classToColor);

    return toReturn;
  }

  public File getInputFile() {
    return this.inputFile;
  }

  public int getSizeMin() {
    return this.sizeMin;
  }

  public int getSizeMax() {
    return this.sizeMax;
  }


  public class ForceAtlasParams {
    private boolean useStrongGravity;
    private double gravity;
    private double scaling;
    private double edgeWeight;

    public boolean getUseStrongGravity() {
      return this.useStrongGravity;
    }

    public double getGravity() {
      return this.gravity;
    }

    public double getScaling() {
      return this.scaling;
    }

    public double getEdgeWeight() {
      return this.edgeWeight;
    }

    public ForceAtlasParams() {
      this.useStrongGravity = true;
      this.gravity = .1;
      this.edgeWeight = .4;
      this.scaling = 10.0;
    }

    public String toString() {
      String toReturn = "Strong: " + useStrongGravity + "\n";
      toReturn += "Gravity: " + gravity + "\n";
      toReturn += "Scaling: " + scaling + "\n";
      toReturn += "Edge Weight: " + edgeWeight;
      return  toReturn;
    }
  }
}
