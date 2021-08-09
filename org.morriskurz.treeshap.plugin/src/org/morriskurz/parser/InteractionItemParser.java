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
 * Item Parser for SHAP value interactions.
 *
 * <p>The parser adds the corresponding specs (column names and types) to the data table, as well as
 * adding the generated SHAP interaction values to each row.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public class InteractionItemParser implements PredictionItemParser<ITreeSHAPPrediction> {

  private final String[] columnNames;

  /**
   * Create an explanation item parser.
   *
   * @param trainingDataSpec Data table spec of the training data. This should only be the columns
   *     used in training and the target column.
   */
  public InteractionItemParser(final DataTableSpec trainingDataSpec) {
    // Don't copy the target column.
    columnNames =
        Arrays.copyOf(
            trainingDataSpec.getColumnNames(), trainingDataSpec.getColumnNames().length - 1);
  }

  @Override
  public void appendCells(final List<DataCell> cells, final ITreeSHAPPrediction prediction) {
    java.util.Collections.addAll(cells, prediction.getInteractionValues());
  }

  @Override
  public void appendSpecs(
      final UniqueNameGenerator nameGenerator, final List<DataColumnSpec> specs) {
    final String[] interactionColumnNames = new String[columnNames.length];
    System.arraycopy(columnNames, 0, interactionColumnNames, 0, columnNames.length);
    // interactionColumnNames[columnNames.length] = "Bias";
    for (final String columnName : interactionColumnNames) {
      for (final String interactedColumnName : interactionColumnNames) {
        String colName = "SHAP " + columnName + " " + interactedColumnName;
        if (columnName.equals(interactedColumnName)) {
          colName = "SHAP " + columnName;
        }
        specs.add(nameGenerator.newColumn(colName, DoubleCell.TYPE));
      }
    }
  }
}
