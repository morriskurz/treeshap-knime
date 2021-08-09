package org.morriskurz.gradientboosted.regression;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.knime.base.node.mine.treeensemble2.data.PredictorRecord;
import org.knime.base.node.mine.treeensemble2.model.AbstractTreeNode;
import org.knime.base.node.mine.treeensemble2.model.GradientBoostedTreesModel;
import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModel;
import org.knime.base.node.mine.treeensemble2.model.TreeNodeRegression;
import org.knime.base.node.mine.treeensemble2.node.predictor.AbstractPredictor;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.def.DoubleCell;

public class ExplanationGBTRegressionPredictor
    extends AbstractPredictor<ExplanationGBTRegressionPrediction> {

  private class GBTPrediction extends ExplanationGBTRegressionPrediction {

    private final double prediction;

    private final List<Map<AbstractTreeNode, Double>> treeMaps;

    public GBTPrediction(
        final PredictorRecord record,
        final TreeNodeRegression[] roots,
        final int nrColumns,
        final int maxDepth,
        final List<Map<AbstractTreeNode, Double>> treeMaps) {
      super(record);
      prediction = model.predict(record);
      this.treeMaps = treeMaps;
      if (maxDepth != -1) {
        computeShap(roots, nrColumns, 1, maxDepth, 0, 0);
      }
    }

    public GBTPrediction(
        final PredictorRecord record,
        final TreeNodeRegression[] roots,
        final int nrColumns,
        final int maxDepth,
        final List<Map<AbstractTreeNode, Double>> treeMaps,
        final int maxNodes,
        final int[] uniqueFeaturesPerTree) {
      super(record);
      prediction = model.predict(record);
      this.treeMaps = treeMaps;
      if (maxDepth != -1 && uniqueFeaturesPerTree != null) {
        dense_tree_interactions_path_dependent(
            roots, nrColumns, 1, maxDepth, maxNodes, uniqueFeaturesPerTree);
      }
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

    @Override
    public double getPrediction() {
      return prediction;
    }

    @Override
    public DoubleCell[] getSHAPValues() {
      // The initial value is not saved in the regression trees themselves.
      final double[] phi = getPhi();
      final double bias = phi[phi.length - 1];
      phi[phi.length - 1] = bias + model.getInitialValue();
      return super.getSHAPValues();
    }
  }

  private final GradientBoostedTreesModel model;

  private final int nrColumns;

  private final int maxDepth;

  private final boolean calculateInteractions;

  private final int maxNodes;

  private final int[] uniqueFeaturesPerTree;

  private final List<Map<AbstractTreeNode, Double>> treeMaps;

  public ExplanationGBTRegressionPredictor(
      final GradientBoostedTreesModel model,
      final Function<DataRow, PredictorRecord> rowConverter,
      final DataTableSpec learnSpec,
      final List<Map<AbstractTreeNode, Double>> treeMaps,
      final int maxDepth,
      final int maxNodes,
      final int[] uniqueFeaturesPerTree) {
    super(rowConverter);
    this.model = model;
    nrColumns = learnSpec.getNumColumns() - 1;
    this.maxDepth = maxDepth;
    this.treeMaps = treeMaps;
    calculateInteractions = uniqueFeaturesPerTree != null;
    this.maxNodes = maxNodes;
    this.uniqueFeaturesPerTree = uniqueFeaturesPerTree;
  }

  private TreeNodeRegression[] initRoots(final TreeEnsembleModel model) {
    final TreeNodeRegression[] roots = new TreeNodeRegression[model.getNrModels()];
    for (int i = 0; i < model.getNrModels(); i++) {
      roots[i] = model.getTreeModelRegression(i).getRootNode();
    }
    return roots;
  }

  @Override
  protected ExplanationGBTRegressionPrediction predictRecord(final PredictorRecord record) {
    if (calculateInteractions) {
      return new GBTPrediction(
          record, initRoots(model), nrColumns, maxDepth, treeMaps, maxNodes, uniqueFeaturesPerTree);
    }
    return new GBTPrediction(record, initRoots(model), nrColumns, maxDepth, treeMaps);
  }
}
