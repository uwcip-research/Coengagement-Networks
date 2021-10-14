public class ClassCount implements Comparable<ClassCount> {
  private int modClass;
  private int count;

  // A tiny object which stores a label for a class and a count of the number of occurrences,
  // with a custom sorting function defined for ease of use in custom coloring the graph.
  public ClassCount(int modClass) {
    this.modClass = modClass;
    this.count = 0;
  }

  public void increment() {
    this.count += 1;
  }

  public int getModClass() {
    return this.modClass;
  }

  public int getCount() {
    return this.count;
  }

  @Override
  public int compareTo(ClassCount o) {
    return - (this.count - o.count);
  }
}
