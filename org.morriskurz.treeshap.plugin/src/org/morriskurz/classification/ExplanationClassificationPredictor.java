/** */
package org.morriskurz.classification;

import org.knime.base.node.mine.treeensemble2.data.PredictorRecord;
import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModel;
import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModelPortObjectSpec;
import org.knime.base.node.mine.treeensemble2.model.TreeModelClassification;
import org.knime.base.node.mine.treeensemble2.model.TreeNodeClassification;
import org.knime.base.node.mine.treeensemble2.node.predictor.AbstractRandomForestPredictor;
import org.knime.base.node.mine.treeensemble2.node.predictor.classification.Voting;
import org.knime.base.node.mine.treeensemble2.node.predictor.classification.VotingFactory;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.RowKey;
import org.knime.core.node.InvalidSettingsException;

/**
 * Predictor implementation for a classification Random Forest.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public final class ExplanationClassificationPredictor
    extends AbstractRandomForestPredictor<ExplanationClassificationPrediction> {

  /**
   * Specific implementation of the classification RF prediction interface.
   *
   * @author Morris Kurz, morriskurz@gmail.com
   */
  private class RFClassificationPrediction extends ExplanationClassificationPrediction {

    private final Voting voting;

    final int nrModels;

    final int positiveClassIndex;

    /**
     * Computes the SHAP values.
     *
     * @param record
     * @param key
     * @param hasOutOfBagFilter
     * @param nrColumns
     * @param maxDepth
     * @param positiveClassIndex
     */
    RFClassificationPrediction(
        final PredictorRecord record,
        final RowKey key,
        final boolean hasOutOfBagFilter,
        final int nrColumns,
        final int maxDepth,
        final int positiveClassIndex) {
      super(record, roots, nrColumns, maxDepth);
      voting = votingFactory.createVoting();
      nrModels = m_model.getNrModels();
      for (int i = 0; i < nrModels; i++) {
        if (hasOutOfBagFilter && isRowPartOfTrainingData(key, i)) {
          // ignore, row was used to train the model
        } else {
          final TreeModelClassification m = m_model.getTreeModelClassification(i);
          final TreeNodeClassification match = m.findMatchingNode(record);
          voting.addVote(match);
        }
      }
      this.positiveClassIndex = positiveClassIndex;
    }

    /**
     * Computes the interaction values.
     *
     * @param record
     * @param key
     * @param hasOutOfBagFilter
     * @param nrColumns
     * @param maxDepth
     * @param positiveClassIndex
     */
    RFClassificationPrediction(
        final PredictorRecord record,
        final RowKey key,
        final boolean hasOutOfBagFilter,
        final int nrColumns,
        final int maxDepth,
        final int positiveClassIndex,
        final int maxNodes,
        final int[] uniqueFeaturesPerTree) {
      super(record, roots, nrColumns, maxDepth, maxNodes, uniqueFeaturesPerTree);
      voting = votingFactory.createVoting();
      nrModels = m_model.getNrModels();
      for (int i = 0; i < nrModels; i++) {
        if (hasOutOfBagFilter && isRowPartOfTrainingData(key, i)) {
          // ignore, row was used to train the model
        } else {
          final TreeModelClassification m = m_model.getTreeModelClassification(i);
          final TreeNodeClassification match = m.findMatchingNode(record);
          voting.addVote(match);
        }
      }
      this.positiveClassIndex = positiveClassIndex;
    }

    /*
     * (non-Javadoc)
     *
     * @see org.knime.base.node.mine.treeensemble2.node.predictor.
     * ClassificationPrediction#getClassPrediction()
     */
    @Override
    public String getClassPrediction() {
      return voting.getMajorityClass();
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
      return voting.getNrVotes();
    }

    @Override
    protected double getNodeValue(final TreeNodeClassification node, final int treeIndex) {
      return node.getTargetDistribution()[positiveClassIndex] / sum(node.getTargetDistribution());
    }

    @Override
    protected double getNumberOfSamples(final TreeNodeClassification node) {
      return sum(node.getTargetDistribution());
    }

    /*
     * (non-Javadoc)
     *
     * @see org.knime.base.node.mine.treeensemble2.node.predictor.
     * ClassificationPrediction#getProbability(int)
     */
    @Override
    public double getProbability(final int classIdx) {
      return voting.getClassProbabilityForClass(classIdx);
    }

    /*
     * (non-Javadoc)
     *
     * @see org.knime.base.node.mine.treeensemble2.node.predictor.
     * ClassificationPrediction#getWinningClassIdx()
     */
    @Override
    public int getWinningClassIdx() {
      return voting.getMajorityClassIdx();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasPrediction() {
      return voting.getNrVotes() > 0;
    }

    private float sum(final float[] a) {
      float result = 0;
      for (final float temp : a) {
        result += temp;
      }
      return result;
    }
  }

  private final VotingFactory votingFactory;

  private final TreeNodeClassification[] roots;

  private final int nrColumns;

  private final int maxDepth;

  private final int positiveClassIndex;

  private final boolean calculateInteractions;

  private final int maxNodes;

  private final int[] uniqueFeaturesPerTree;

  /**
   * @param model
   * @param modelSpec
   * @param predictSpec
   * @param votingFactory
   * @param quantileBinning
   * @throws InvalidSettingsException
   */
  public ExplanationClassificationPredictor(
      final TreeEnsembleModel model,
      final TreeEnsembleModelPortObjectSpec modelSpec,
      final DataTableSpec predictSpec,
      final VotingFactory votingFactory,
      final int maxDepth,
      final int positiveClassIndex,
      final int maxNodes,
      final int[] uniqueFeaturesPerTree)
      throws InvalidSettingsException {
    super(model, modelSpec, predictSpec);
    this.votingFactory = votingFactory;
    this.maxDepth = maxDepth;
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
    this.positiveClassIndex = positiveClassIndex;
    calculateInteractions = uniqueFeaturesPerTree != null;
    this.maxNodes = maxNodes;
    this.uniqueFeaturesPerTree = uniqueFeaturesPerTree;
  }

  private TreeNodeClassification[] initRoots(final TreeEnsembleModel model) {
    final TreeNodeClassification[] rootsTemp = new TreeNodeClassification[model.getNrModels()];
    for (int i = 0; i < model.getNrModels(); i++) {
      rootsTemp[i] = model.getTreeModelClassification(i).getRootNode();
    }
    return rootsTemp;
  }

  /*
   * (non-Javadoc)
   *
   * @see org.knime.base.node.mine.treeensemble2.node.predictor.
   * AbstractRandomForestPredictor#predictRecord(org.knime.base.node.mine.
   * treeensemble2.data.PredictorRecord, org.knime.core.data.RowKey)
   */
  @Override
  public ExplanationClassificationPrediction predictRecord(
      final PredictorRecord record, final RowKey key) {
    if (calculateInteractions) {
      return new RFClassificationPrediction(
          record,
          key,
          hasOutOfBagFilter(),
          nrColumns,
          maxDepth,
          positiveClassIndex,
          maxNodes,
          uniqueFeaturesPerTree);
    }
    return new RFClassificationPrediction(
        record, key, hasOutOfBagFilter(), nrColumns, maxDepth, positiveClassIndex);
  }
}
