/** */
package org.morriskurz.classification;

import org.knime.base.node.mine.treeensemble2.data.PredictorRecord;
import org.knime.base.node.mine.treeensemble2.model.TreeNodeClassification;
import org.knime.base.node.mine.treeensemble2.node.predictor.RandomForestClassificationPrediction;
import org.morriskurz.parser.TreeSHAPPrediction;

/**
 * Prediction class for a classification RF prediction implementing the explanation interface.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public abstract class ExplanationClassificationPrediction
    extends TreeSHAPPrediction<TreeNodeClassification>
    implements RandomForestClassificationPrediction {

  /**
   * Computes the SHAP values.
   *
   * @param record
   * @param roots
   * @param nrColumns
   * @param maxDepth
   */
  protected ExplanationClassificationPrediction(
      final PredictorRecord record,
      final TreeNodeClassification[] roots,
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
  protected ExplanationClassificationPrediction(
      final PredictorRecord record,
      final TreeNodeClassification[] roots,
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
