/** */
package org.morriskurz.regression;

import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.commons.math.stat.descriptive.moment.Variance;
import org.knime.base.node.mine.treeensemble2.data.PredictorRecord;
import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModel;
import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModelPortObjectSpec;
import org.knime.base.node.mine.treeensemble2.model.TreeModelRegression;
import org.knime.base.node.mine.treeensemble2.model.TreeNodeRegression;
import org.knime.base.node.mine.treeensemble2.node.predictor.AbstractRandomForestPredictor;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.RowKey;
import org.knime.core.node.InvalidSettingsException;

/** @author Morris Kurz, morriskurz@gmail.com */
public class ExplanationRegressionPredictor
    extends AbstractRandomForestPredictor<ExplanationRegressionPrediction> {

  private class RFRegressionPrediction extends ExplanationRegressionPrediction {

    private final double m_mean;

    private final double[] m_mean_per_tree;

    private final double m_variance;

    private final int m_modelCount;

    RFRegressionPrediction(
        final PredictorRecord record,
        final RowKey key,
        final boolean hasOutOfBagFilter,
        final int maxDepth) {
      super(record, roots, nrColumns, maxDepth);
      final Mean mean = new Mean();
      final Variance variance = new Variance();
      final int nrModels = m_model.getNrModels();
      m_mean_per_tree = new double[nrModels];
      for (int i = 0; i < nrModels; i++) {
        if (hasOutOfBagFilter && isRowPartOfTrainingData(key, i)) {
          // ignore, row was used to train the model
        } else {
          final TreeModelRegression m = m_model.getTreeModelRegression(i);
          final TreeNodeRegression match = m.findMatchingNode(record);
          final double nodeMean = match.getMean();
          m_mean_per_tree[i] = nodeMean;
          mean.increment(nodeMean);
          variance.increment(nodeMean);
        }
      }
      m_modelCount = (int) mean.getN();
      m_variance = variance.getResult();
      m_mean = mean.getResult();
    }

    /**
     * Computes the interaction values.
     *
     * @param record
     * @param key
     * @param hasOutOfBagFilter
     * @param maxDepth
     * @param maxNodes
     * @param uniqueFeaturesPerTree
     */
    RFRegressionPrediction(
        final PredictorRecord record,
        final RowKey key,
        final boolean hasOutOfBagFilter,
        final int maxDepth,
        final int maxNodes,
        final int[] uniqueFeaturesPerTree) {
      super(record, roots, nrColumns, maxDepth, maxNodes, uniqueFeaturesPerTree);
      final Mean mean = new Mean();
      final Variance variance = new Variance();
      final int nrModels = m_model.getNrModels();
      m_mean_per_tree = new double[nrModels];
      for (int i = 0; i < nrModels; i++) {
        if (hasOutOfBagFilter && isRowPartOfTrainingData(key, i)) {
          // ignore, row was used to train the model
        } else {
          final TreeModelRegression m = m_model.getTreeModelRegression(i);
          final TreeNodeRegression match = m.findMatchingNode(record);
          final double nodeMean = match.getMean();
          m_mean_per_tree[i] = nodeMean;
          mean.increment(nodeMean);
          variance.increment(nodeMean);
        }
      }
      m_modelCount = (int) mean.getN();
      m_variance = variance.getResult();
      m_mean = mean.getResult();
    }

    /*
     * (non-Javadoc)
     *
     * @see
     * org.knime.base.node.mine.treeensemble2.node.predictor.OutOfBagPrediction#
     * getModelCount()
     */
    @Override
    public int getModelCount() {
      return m_modelCount;
    }

    @Override
    protected double getNodeValue(final TreeNodeRegression node, final int treeIndex) {
      return node.getMean();
    }

    @Override
    protected double getNumberOfSamples(final TreeNodeRegression node) {
      return node.getTotalSum();
    }

    /*
     * (non-Javadoc)
     *
     * @see
     * org.knime.base.node.mine.treeensemble2.node.predictor.RegressionPrediction#
     * getPrediction()
     */
    @Override
    public double getPrediction() {
      return m_mean;
    }

    /*
     * (non-Javadoc)
     *
     * @see org.knime.base.node.mine.treeensemble2.node.predictor.
     * RandomForestRegressionPrediction#getVariance()
     */
    @Override
    public double getVariance() {
      return m_variance;
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasPrediction() {
      return m_modelCount != 0;
    }
  }

  private final TreeNodeRegression[] roots;

  private final int nrColumns;

  private final int maxDepth;

  private final boolean calculateInteractions;

  private final int maxNodes;

  private final int[] uniqueFeaturesPerTree;

  /**
   * @param model
   * @param modelSpec
   * @param predictSpec
   * @throws InvalidSettingsException
   */
  public ExplanationRegressionPredictor(
      final TreeEnsembleModel model,
      final TreeEnsembleModelPortObjectSpec modelSpec,
      final DataTableSpec predictSpec,
      final int maxDepth,
      final int maxNodes,
      final int[] uniqueFeaturesPerTree)
      throws InvalidSettingsException {
    super(model, modelSpec, predictSpec);
    if (model != null) {
      roots = initRoots(model);
    } else {
      roots = null;
    }
    if (modelSpec != null) {
      nrColumns = modelSpec.getTableSpec().getNumColumns() - 1;
    } else {
      nrColumns = 0;
    }
    this.maxDepth = maxDepth;
    calculateInteractions = uniqueFeaturesPerTree != null;
    this.maxNodes = maxNodes;
    this.uniqueFeaturesPerTree = uniqueFeaturesPerTree;
  }

  private TreeNodeRegression[] initRoots(final TreeEnsembleModel model) {
    final TreeNodeRegression[] tempRoots = new TreeNodeRegression[model.getNrModels()];
    for (int i = 0; i < model.getNrModels(); i++) {
      tempRoots[i] = model.getTreeModelRegression(i).getRootNode();
    }
    return tempRoots;
  }

  /*
   * (non-Javadoc)
   *
   * @see org.knime.base.node.mine.treeensemble2.node.predictor.
   * AbstractRandomForestPredictor#predictRecord(org.knime.base.node.mine.
   * treeensemble2.data.PredictorRecord, org.knime.core.data.RowKey)
   */
  @Override
  public ExplanationRegressionPrediction predictRecord(
      final PredictorRecord record, final RowKey key) {
    if (calculateInteractions) {
      return new RFRegressionPrediction(
          record, key, hasOutOfBagFilter(), maxDepth, maxNodes, uniqueFeaturesPerTree);
    }
    return new RFRegressionPrediction(record, key, hasOutOfBagFilter(), maxDepth);
  }
}
