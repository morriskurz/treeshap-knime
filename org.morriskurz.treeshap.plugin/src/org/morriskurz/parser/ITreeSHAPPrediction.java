package org.morriskurz.parser;

import org.knime.base.node.mine.treeensemble2.node.predictor.Prediction;
import org.knime.core.data.def.DoubleCell;

/**
 * Interface for a SHAP prediction.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public interface ITreeSHAPPrediction extends Prediction {
  DoubleCell[] getInteractionValues();

  DoubleCell[] getSHAPValues();
}
