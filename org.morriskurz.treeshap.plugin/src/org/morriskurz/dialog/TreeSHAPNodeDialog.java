package org.morriskurz.dialog;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.port.PortObjectSpec;

/**
 * The node dialog for the classification RF TreeSHAP node.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public final class TreeSHAPNodeDialog extends NodeDialogPane {

  private final ExplainerPanel predictorPanel;

  /** */
  public TreeSHAPNodeDialog(final boolean isRegression, final boolean isRandomForest) {
    predictorPanel = new ExplainerPanel(isRegression, isRandomForest);
    addTab(ExplainerPanel.PANEL_NAME, predictorPanel);
  }

  /** {@inheritDoc} */
  @Override
  protected void loadSettingsFrom(final NodeSettingsRO settings, final PortObjectSpec[] specs)
      throws NotConfigurableException {
    predictorPanel.loadSettingsFrom(settings, specs);
  }

  /** {@inheritDoc} */
  @Override
  protected void saveSettingsTo(final NodeSettingsWO settings) throws InvalidSettingsException {
    predictorPanel.saveSettingsTo(settings);
  }
}
