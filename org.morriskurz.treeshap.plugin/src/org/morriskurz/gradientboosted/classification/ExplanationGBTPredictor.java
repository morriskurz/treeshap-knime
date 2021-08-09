package org.morriskurz.gradientboosted.classification;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.knime.base.node.mine.treeensemble2.data.PredictorRecord;
import org.knime.base.node.mine.treeensemble2.model.AbstractTreeNode;
import org.knime.base.node.mine.treeensemble2.model.MultiClassGradientBoostedTreesModel;
import org.knime.base.node.mine.treeensemble2.model.TreeNodeRegression;
import org.knime.base.node.mine.treeensemble2.node.predictor.AbstractPredictor;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.def.DoubleCell;

public class ExplanationGBTPredictor extends AbstractPredictor<ExplanationGBTPrediction> {

  private class GBTPrediction extends ExplanationGBTPrediction {

    private final int winningClassIdx;

    private final double[] probabilities;

    private final List<Map<AbstractTreeNode, Double>> treeMaps;

    private final int positiveClassIndex;

    public GBTPrediction(
        final PredictorRecord record,
        final TreeNodeRegression[] roots,
        final int nrColumns,
        final int winningClassIdx,
        final int maxDepth,
        final double[] probabilities,
        final List<Map<AbstractTreeNode, Double>> treeMaps,
        final int positiveClassIndex) {
      super(record, roots, nrColumns, maxDepth);
      this.winningClassIdx = winningClassIdx;
      this.probabilities = probabilities;
      this.treeMaps = treeMaps;
      this.positiveClassIndex = positiveClassIndex;
      if (maxDepth != -1) {
        computeShap(roots, nrColumns, 1, maxDepth, 0, 0);
      }
    }

    public GBTPrediction(
        final PredictorRecord record,
        final TreeNodeRegression[] roots,
        final int nrColumns,
        final int winningClassIdx,
        final int maxDepth,
        final double[] probabilities,
        final List<Map<AbstractTreeNode, Double>> treeMaps,
        final int positiveClassIndex,
        final int maxNodes,
        final int[] uniqueFeaturesPerTree) {
      super(record, roots, nrColumns, maxDepth);
      this.winningClassIdx = winningClassIdx;
      this.probabilities = probabilities;
      this.treeMaps = treeMaps;
      this.positiveClassIndex = positiveClassIndex;
      if (maxDepth != -1 && uniqueFeaturesPerTree != null) {
        dense_tree_interactions_path_dependent(
            roots, nrColumns, 1, maxDepth, maxNodes, uniqueFeaturesPerTree);
      }
    }

    private double expit(final double value) {
      return 1.0f / (1 + Math.exp(-value));
    }

    /*
     * (non-Javadoc)
     *
     * @see org.knime.base.node.mine.treeensemble2.node.predictor.
     * ClassificationPrediction#getClassPrediction()
     */
    @Override
    public String getClassPrediction() {
      return model.getClassLabel(winningClassIdx);
    }

    @Override
    protected double getNodeValue(final TreeNodeRegression node, final int treeIndex) {
      final Map<AbstractTreeNode, Double> nodeToValue = treeMaps.get(treeIndex);
      return nodeToValue.get(node);
    }

    @Override
    protected double getNumberOfSamples(final TreeNodeRegression node) {
      return node.getTotalSum();
    }

    /*
     * (non-Javadoc)
     *
     * @see org.knime.base.node.mine.treeensemble2.node.predictor.
     * ClassificationPrediction#getProbability(int)
     */
    @Override
    public double getProbability(final int classIdx) {
      assert probabilities != null
          : "No probabilities were calculated. This is likely a coding issue.";
      return probabilities[classIdx];
    }

    @Override
    public DoubleCell[] getSHAPValues() {
      // Transform from logit to probability by scaling. This is not exact,
      // but the best one can do.
      final double[] phi = getPhi();
      // TODO: For now, only the raw logits are output because the probability scaling
      // is not sufficient.
      /*double absoluteSum = 0;
      // Last entry is the BIAS, not relevant for the scaling
      for (int i = 0; i < phi.length - 1; i++) {
        absoluteSum += phi[i];
      }
      absoluteSum = Math.abs(absoluteSum);
      // Transform bias to probability
      phi[phi.length - 1] = expit(phi[phi.length - 1]);
      System.out.println(phi[phi.length - 1]);
      if (absoluteSum <= Double.MIN_VALUE) {
        // Should not happen often, but here the scaling can get arbitrary.
        // Scale it such that the highest SHAP value is 0.25
        for (int i = 0; i < phi.length - 1; i++) {
          absoluteSum = Math.max(absoluteSum, Math.abs(phi[i]) * 4);
        }
      }
      final double transformedTargetProbability = probabilities[positiveClassIndex];
      final double difference = Math.abs(transformedTargetProbability - phi[phi.length - 1]);
      for (int i = 0; i < phi.length - 1; i++) {
        phi[i] = phi[i] / absoluteSum * difference;
      }*/
      final DoubleCell[] newCells = new DoubleCell[phi.length];
      for (int i = 0; i < phi.length; i++) {
        newCells[i] = new DoubleCell(phi[i]);
      }
      return newCells;
    }

    /*
     * (non-Javadoc)
     *
     * @see org.knime.base.node.mine.treeensemble2.node.predictor.
     * ClassificationPrediction#getWinningClassIdx()
     */
    @Override
    public int getWinningClassIdx() {
      return winningClassIdx;
    }
  }

  private static int argmax(final double[] vector) {
    int argmax = -1;
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < vector.length; i++) {
      if (vector[i] > max) {
        max = vector[i];
        argmax = i;
      }
    }
    return argmax;
  }

  private static double max(final double[] logits) {
    double max = Double.NEGATIVE_INFINITY;
    for (final double logit : logits) {
      if (logit > max) {
        max = logit;
      }
    }
    return max;
  }

  private final MultiClassGradientBoostedTreesModel model;

  private final int nrColumns;

  private final int maxDepth;

  private final List<Map<AbstractTreeNode, Double>> treeMaps;

  private final int positiveClassIndex;

  private final boolean calculateInteractions;

  private final int maxNodes;

  private final int[] uniqueFeaturesPerTree;

  /**
   * Constructor for classification gbt predictors.
   *
   * @param model the gradient boosted trees model
   * @param rowConverter converts input {@link DataRow rows} into {@link PredictorRecord records}
   * @param learnSpec data table spec of the learning table without unnecessary columns
   * @param calculateProbabilities indicates whether probabilities should be calculated
   * @param positiveClassIndex Index of the positive class. Has to be between 0 and nrClasses-1.
   */
  public ExplanationGBTPredictor(
      final MultiClassGradientBoostedTreesModel model,
      final Function<DataRow, PredictorRecord> rowConverter,
      final DataTableSpec learnSpec,
      final boolean calculateProbabilities,
      final int positiveClassIndex,
      final int maxDepth,
      final List<Map<AbstractTreeNode, Double>> trees,
      final int maxNodes,
      final int[] uniqueFeaturesPerTree) {
    super(rowConverter);
    this.model = model;
    nrColumns = learnSpec.getNumColumns() - 1;
    this.maxDepth = maxDepth;
    treeMaps = trees;
    this.positiveClassIndex = positiveClassIndex;
    calculateInteractions = uniqueFeaturesPerTree != null;
    this.maxNodes = maxNodes;
    this.uniqueFeaturesPerTree = uniqueFeaturesPerTree;
  }

  private double[] calculateLogits(final PredictorRecord record) {
    final int nrClasses = model.getNrClasses();
    final int nrLevels = model.getNrLevels();
    final double[] logits = new double[nrClasses];
    Arrays.fill(logits, model.getInitialValue());
    for (int i = 0; i < nrLevels; i++) {
      for (int j = 0; j < nrClasses; j++) {
        final TreeNodeRegression matchingNode = model.getModel(i, j).findMatchingNode(record);
        logits[j] += model.getCoefficientMap(i, j).get(matchingNode.getSignature());
      }
    }
    return logits;
  }

  private double getNormalizationConstant(final double[] logits) {
    return max(logits);
  }

  private TreeNodeRegression[] initRoots(
      final MultiClassGradientBoostedTreesModel model, final int positiveClassIndex) {
    final TreeNodeRegression[] roots = new TreeNodeRegression[model.getNrLevels()];
    // Note that GBT has one regression tree per class, doesn't fit
    // with the usual models.
    for (int i = 0; i < model.getNrLevels(); i++) {
      roots[i] = model.getModel(i, positiveClassIndex).getRootNode();
    }
    return roots;
  }

  @Override
  protected ExplanationGBTPrediction predictRecord(final PredictorRecord record) {
    final double[] logits = calculateLogits(record);
    transformToProbabilities(logits);
    if (calculateInteractions) {
      return new GBTPrediction(
          record,
          initRoots(model, positiveClassIndex),
          nrColumns,
          argmax(logits),
          maxDepth,
          logits,
          treeMaps,
          positiveClassIndex,
          maxNodes,
          uniqueFeaturesPerTree);
    }
    return new GBTPrediction(
        record,
        initRoots(model, positiveClassIndex),
        nrColumns,
        argmax(logits),
        maxDepth,
        logits,
        treeMaps,
        positiveClassIndex);
  }

  private void transformToProbabilities(final double[] logits) {
    final double[] probabilities = logits;
    double expSum = 0;
    final double constant = getNormalizationConstant(logits);
    for (int i = 0; i < logits.length; i++) {
      final double exp = Math.exp(logits[i] - constant);
      probabilities[i] = exp;
      expSum += exp;
    }
    assert expSum > 0 : "The exponential sum was zero.";
    for (int i = 0; i < probabilities.length; i++) {
      probabilities[i] /= expSum;
    }
  }
}
