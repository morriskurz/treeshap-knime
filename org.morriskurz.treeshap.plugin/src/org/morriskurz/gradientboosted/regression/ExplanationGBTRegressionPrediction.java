package org.morriskurz.gradientboosted.regression;

import org.knime.base.node.mine.treeensemble2.data.PredictorRecord;
import org.knime.base.node.mine.treeensemble2.model.TreeNodeRegression;
import org.knime.base.node.mine.treeensemble2.node.predictor.RegressionPrediction;
import org.morriskurz.parser.TreeSHAPPrediction;

/** @author Morris Kurz, morriskurz@gmail.com */
public abstract class ExplanationGBTRegressionPrediction extends TreeSHAPPrediction<TreeNodeRegression>
		implements RegressionPrediction {

	protected ExplanationGBTRegressionPrediction(final PredictorRecord record) {
		super(record);
	}
}
