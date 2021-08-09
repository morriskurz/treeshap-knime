package org.morriskurz.parser;

import java.util.Arrays;
import java.util.List;

import org.knime.base.node.mine.treeensemble2.node.predictor.parser.PredictionItemParser;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.def.DoubleCell;
import org.knime.core.util.UniqueNameGenerator;

/**
 * Item Parser for SHAP value explanations.
 *
 * The parser adds the corresponding specs (column names and types) to the data
 * table, as well as adding the generated SHAP values to each row.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 *
 */
public class ExplanationItemParser implements PredictionItemParser<ITreeSHAPPrediction> {

	private final String[] columnNames;

	/**
	 * Create an explanation item parser.
	 *
	 * @param trainingDataSpec Data table spec of the training data. This should
	 *                         only be the columns used in training and the target
	 *                         column.
	 */
	public ExplanationItemParser(final DataTableSpec trainingDataSpec) {
		// Don't copy the target column.
		columnNames = Arrays.copyOf(trainingDataSpec.getColumnNames(), trainingDataSpec.getColumnNames().length - 1);
	}

	@Override
	public void appendCells(final List<DataCell> cells, final ITreeSHAPPrediction prediction) {
		java.util.Collections.addAll(cells, prediction.getSHAPValues());
	}

	@Override
	public void appendSpecs(final UniqueNameGenerator nameGenerator, final List<DataColumnSpec> specs) {
		for (final String columnName : columnNames) {
			final String colName = "SHAP " + columnName;
			specs.add(nameGenerator.newColumn(colName, DoubleCell.TYPE));
		}
		specs.add(nameGenerator.newColumn("Bias", DoubleCell.TYPE));
	}

}
