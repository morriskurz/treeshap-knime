package org.morriskurz;

import org.knime.base.node.mine.treeensemble2.node.predictor.TreeEnsemblePredictorConfiguration;
import org.knime.core.data.DataCell;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.NotConfigurableException;

/**
 * Configuration class for all TreeSHAP algorithms.
 *
 * <p>It is used for saving and loading model configurations chosen by the user. Specifically, if
 * the model is a regression model, the target column name, and other information used by the
 * classification / regression model.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public class TreeSHAPConfiguration extends TreeEnsemblePredictorConfiguration {

  private static final String CFG_POSITIVE_CLASS = "positiveClass";
  private static final String CFG_SHOW_EXPLANATION = "showExplanation";
  private static final String CFG_COMPUTE_INTERACTIONS = "computeInteractions";

  public static TreeSHAPConfiguration createDefault(
      final boolean isRegression, final String targetColName) {
    return new TreeSHAPConfiguration(isRegression, targetColName);
  }

  private DataCell positiveClass = null;
  private boolean showExplanation = true;
  private String licenseUrl = null;
  private boolean computeInteractions = false;

  public TreeSHAPConfiguration(final boolean isRegression, final String targetColName) {
    super(isRegression, targetColName);
  }

  public String getLicenseUrl() {
    return licenseUrl;
  }

  public DataCell getPositiveClass() {
    return positiveClass;
  }

  @Override
  public void internalLoadInDialog(final NodeSettingsRO settings) throws NotConfigurableException {
    super.internalLoadInDialog(settings);
    positiveClass = settings.getDataCell(CFG_POSITIVE_CLASS, null);
    showExplanation = settings.getBoolean(CFG_SHOW_EXPLANATION, true);
    computeInteractions = settings.getBoolean(CFG_COMPUTE_INTERACTIONS, false);
  }

  @Override
  public void internalLoadInModel(final NodeSettingsRO settings) throws InvalidSettingsException {
    super.internalLoadInModel(settings);
    positiveClass = settings.getDataCell(CFG_POSITIVE_CLASS, null);
    showExplanation = settings.getBoolean(CFG_SHOW_EXPLANATION, true);
    computeInteractions = settings.getBoolean(CFG_COMPUTE_INTERACTIONS, false);
  }

  @Override
  public void internalSave(final NodeSettingsWO settings) {
    super.internalSave(settings);
    settings.addDataCell(CFG_POSITIVE_CLASS, positiveClass);
    settings.addBoolean(CFG_SHOW_EXPLANATION, showExplanation);
    settings.addBoolean(CFG_COMPUTE_INTERACTIONS, computeInteractions);
  }

  public boolean isComputeInteractions() {
    return computeInteractions;
  }

  public boolean isShowExplanation() {
    return showExplanation;
  }

  public void setComputeInteractions(final boolean computeInteractions) {
    this.computeInteractions = computeInteractions;
  }

  public void setPositiveClass(final DataCell positiveClass) {
    this.positiveClass = positiveClass;
  }

  public void setShowExplanation(final boolean showExplanation) {
    this.showExplanation = showExplanation;
  }
}
