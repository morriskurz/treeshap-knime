/** */
package org.morriskurz.regression;

import org.knime.base.node.mine.treeensemble2.data.PredictorRecord;
import org.knime.base.node.mine.treeensemble2.model.TreeNodeRegression;
import org.knime.base.node.mine.treeensemble2.node.predictor.RandomForestRegressionPrediction;
import org.morriskurz.parser.TreeSHAPPrediction;

/** @author Morris Kurz, morriskurz@gmail.com */
public abstract class ExplanationRegressionPrediction extends TreeSHAPPrediction<TreeNodeRegression>
    implements RandomForestRegressionPrediction {

  protected ExplanationRegressionPrediction(
      final PredictorRecord record,
      final TreeNodeRegression[] roots,
      final int nrColumns,
      final int maxDepth) {
    super(record);
    if (maxDepth != -1) {
      computeShap(roots, nrColumns, 1.0f / roots.length, maxDepth, 0, 0);
    }
  }

  /**
   * Computes the interaction values.
   *
   * @param record
   * @param roots
   * @param nrColumns
   * @param maxDepth
   * @param maxNodes
   * @param uniqueFeaturesPerTree
   */
  protected ExplanationRegressionPrediction(
      final PredictorRecord record,
      final TreeNodeRegression[] roots,
      final int nrColumns,
      final int maxDepth,
      final int maxNodes,
      final int[] uniqueFeaturesPerTree) {
    super(record);
    if (maxDepth != -1) {
      dense_tree_interactions_path_dependent(
          roots, nrColumns, 1.0f / roots.length, maxDepth, maxNodes, uniqueFeaturesPerTree);
    }
  }
}
