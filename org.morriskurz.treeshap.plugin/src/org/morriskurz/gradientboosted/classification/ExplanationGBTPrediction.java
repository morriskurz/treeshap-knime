package org.morriskurz.gradientboosted.classification;

import org.knime.base.node.mine.treeensemble2.data.PredictorRecord;
import org.knime.base.node.mine.treeensemble2.model.TreeNodeRegression;
import org.knime.base.node.mine.treeensemble2.node.predictor.ClassificationPrediction;
import org.morriskurz.parser.TreeSHAPPrediction;

/** @author Morris Kurz, morriskurz@gmail.com */
public abstract class ExplanationGBTPrediction extends TreeSHAPPrediction<TreeNodeRegression>
		implements ClassificationPrediction {

	protected ExplanationGBTPrediction(final PredictorRecord record, final TreeNodeRegression[] roots,
			final int nrColumns, final int maxDepth) {
		super(record);
	}
}
