package org.morriskurz.parser;

/**
 * A path element of a decision tree path.
 *
 * <p>It saves four values, the feature index which is split on, the fraction of "zero" paths (where
 * this feature is not in the set S) and the fraction of "one" paths (where this feature is in the
 * set S). The proportion of sets S of a given cardinality that are present is used in the original
 * code, but omitted here because of how array copies work in Java.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public class PathElement {

  private int featureIndexForSplit;
  private double fractionOfZeroPaths;
  private double fractionOfOnePaths;

  public PathElement(
      final int featureIndexForSplit,
      final double fractionOfZeroPaths,
      final double fractionOfOnePaths) {
    super();
    this.featureIndexForSplit = featureIndexForSplit;
    this.fractionOfZeroPaths = fractionOfZeroPaths;
    this.fractionOfOnePaths = fractionOfOnePaths;
  }

  public int getFeatureIndexForSplit() {
    return featureIndexForSplit;
  }

  public double getFractionOfOnePaths() {
    return fractionOfOnePaths;
  }

  public double getFractionOfZeroPaths() {
    return fractionOfZeroPaths;
  }

  public void setFeatureIndexForSplit(final int featureIndexForSplit) {
    this.featureIndexForSplit = featureIndexForSplit;
  }

  public void setFractionOfOnePaths(final double fractionOfOnePaths) {
    this.fractionOfOnePaths = fractionOfOnePaths;
  }

  public void setFractionOfZeroPaths(final double fractionOfZeroPaths) {
    this.fractionOfZeroPaths = fractionOfZeroPaths;
  }
}
