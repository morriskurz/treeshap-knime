package org.morriskurz.dialog;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModelPortObjectSpec;
import org.knime.base.node.mine.treeensemble2.node.predictor.TreeEnsemblePredictorConfiguration;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.port.PortObjectSpec;
import org.morriskurz.TreeSHAPConfiguration;

/**
 * Basic panel implementation for the TreeSHAP node.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
@SuppressWarnings("serial")
public final class ExplainerPanel extends JPanel {

  /** Panel name. */
  public static final String PANEL_NAME = "Prediction Settings";

  private final JCheckBox m_appendOverallConfidenceColChecker;

  private final JCheckBox m_appendClassProbabilitiesColChecker;

  private final JTextField m_suffixForClassProbabilitiesTextField;

  private final JLabel m_suffixLabel;

  private final JTextField m_predictionColNameField;

  private final JCheckBox m_changePredictionColNameChecker;

  private final JCheckBox m_useSoftVotingChecker;

  private boolean m_isRegression;

  private final JCheckBox m_showExplanationChecker;

  private final JCheckBox m_computeInteractionsChecker;

  private final boolean m_isRandomForest;


  private final JComboBox<DataCell> m_positiveClass =
      new JComboBox<>(new DefaultComboBoxModel<DataCell>());

  /**
   * @param isRegression panel for regression or classification.
   * @param isRandomForest for random forest type algorithms show the soft voting option
   */
  public ExplainerPanel(final boolean isRegression, final boolean isRandomForest) {
    super(new GridBagLayout());
    m_isRegression = isRegression;
    m_isRandomForest = isRandomForest;
    m_predictionColNameField = new JTextField(20);
    final String defColName = TreeEnsemblePredictorConfiguration.getDefPredictColumnName();
    m_predictionColNameField.setText(defColName);
    m_predictionColNameField.addFocusListener(
        new FocusAdapter() {
          /** {@inheritDoc} */
          @Override
          public void focusGained(final FocusEvent e) {
            if (m_predictionColNameField.getText().equals(defColName)) {
              m_predictionColNameField.selectAll();
            }
          }
        });
    m_changePredictionColNameChecker = new JCheckBox("Change prediction column name");
    m_changePredictionColNameChecker.doClick();
    m_changePredictionColNameChecker.addChangeListener(
        new ChangeListener() {

          @Override
          public void stateChanged(final ChangeEvent e) {
            final JCheckBox source = (JCheckBox) e.getSource();
            m_predictionColNameField.setEnabled(source.isSelected());
          }
        });
    m_appendClassProbabilitiesColChecker = new JCheckBox("Append individual class probabilities");
    m_suffixForClassProbabilitiesTextField = new JTextField(20);
    m_suffixLabel = new JLabel("Suffix for probability columns");
    m_suffixLabel.setEnabled(false);
    m_suffixForClassProbabilitiesTextField.setEnabled(false);
    m_appendClassProbabilitiesColChecker.addChangeListener(
        new ChangeListener() {

          @Override
          public void stateChanged(final ChangeEvent e) {
            final JCheckBox source = (JCheckBox) e.getSource();
            m_suffixForClassProbabilitiesTextField.setEnabled(source.isSelected());
            m_suffixLabel.setEnabled(source.isSelected());
          }
        });
    m_appendOverallConfidenceColChecker = new JCheckBox("Append overall prediction confidence");
    m_useSoftVotingChecker = new JCheckBox("Use soft voting");
    m_showExplanationChecker = new JCheckBox("Show explanation");
    m_showExplanationChecker.addChangeListener(
        new ChangeListener() {

          @Override
          public void stateChanged(final ChangeEvent e) {
            final JCheckBox source = (JCheckBox) e.getSource();
            m_computeInteractionsChecker.setEnabled(source.isSelected());
            m_positiveClass.setEnabled(source.isSelected());
          }
        });
    m_computeInteractionsChecker = new JCheckBox("Compute interactions");
    initLayout();
  }

  /** */
  private void initLayout() {
    final GridBagConstraints gbc = new GridBagConstraints();
    gbc.insets = new Insets(5, 5, 5, 5);
    gbc.anchor = GridBagConstraints.WEST;

    gbc.gridx = 0;
    gbc.gridy = 0;
    add(m_changePredictionColNameChecker, gbc);
    gbc.gridy += 1;
    add(new JLabel("Prediction column name"), gbc);
    gbc.gridx += 1;
    add(m_predictionColNameField, gbc);
    if (!m_isRegression) {
      gbc.gridy += 1;
      gbc.gridx = 0;
      gbc.gridwidth = 2;
      add(m_appendOverallConfidenceColChecker, gbc);

      gbc.gridy += 1;
      gbc.gridx = 0;
      gbc.gridwidth = 2;
      add(m_appendClassProbabilitiesColChecker, gbc);

      gbc.gridy += 1;
      gbc.gridx = 0;
      gbc.gridwidth = 1;
      add(m_suffixLabel, gbc);
      gbc.gridx += 1;
      add(m_suffixForClassProbabilitiesTextField, gbc);
      gbc.gridx = 0;
      if (m_isRandomForest) {
        gbc.gridy += 1;
        gbc.gridx = 0;
        gbc.gridwidth = 2;
        add(m_useSoftVotingChecker, gbc);
      }
      gbc.gridy += 1;
      add(m_showExplanationChecker, gbc);
      gbc.gridy += 1;
      add(m_computeInteractionsChecker, gbc);

      gbc.gridy += 1;
      add(new JLabel("Positive class"), gbc);
      gbc.gridx += 1;
      add(m_positiveClass, gbc);

    } else {
      gbc.gridy += 1;
      gbc.gridx = 0;
      add(m_showExplanationChecker, gbc);
      gbc.gridy += 1;
      add(m_computeInteractionsChecker, gbc);
    }
  }

  /**
   * Loads the settings from the provided <b>settings</b>
   *
   * @param settings
   * @param specs
   * @throws NotConfigurableException
   */
  public void loadSettingsFrom(final NodeSettingsRO settings, final PortObjectSpec[] specs)
      throws NotConfigurableException {
    final TreeEnsembleModelPortObjectSpec treeSpec = (TreeEnsembleModelPortObjectSpec) specs[0];
    try {
      treeSpec.assertTargetTypeMatches(m_isRegression);
    } catch (final InvalidSettingsException e) {
      m_isRegression = !m_isRegression;
    }
    final DataColumnSpec targetColumn = treeSpec.getTargetColumn();
    final TreeSHAPConfiguration config =
        new TreeSHAPConfiguration(m_isRegression, targetColumn.getName());
    config.loadInDialog(settings);
    if (!m_isRegression) {
      m_positiveClass.removeAllItems();
      for (final DataCell cell : targetColumn.getDomain().getValues()) {
        m_positiveClass.addItem(cell);
      }
      if (config.isAppendPredictionConfidence()
          != m_appendOverallConfidenceColChecker.isSelected()) {
        m_appendOverallConfidenceColChecker.doClick();
      }
      m_suffixForClassProbabilitiesTextField.setText(config.getSuffixForClassProbabilities());
      if (config.isAppendClassConfidences() != m_appendClassProbabilitiesColChecker.isSelected()) {
        m_appendClassProbabilitiesColChecker.doClick();
      }
      m_useSoftVotingChecker.setSelected(config.isUseSoftVoting());
      m_positiveClass.setSelectedItem(config.getPositiveClass());
    }
    String colName = config.getPredictionColumnName();
    if (colName == null || colName.isEmpty()) {
      colName = TreeEnsemblePredictorConfiguration.getDefPredictColumnName();
    }
    m_predictionColNameField.setText(colName);
    if (config.isChangePredictionColumnName() != m_changePredictionColNameChecker.isSelected()) {
      m_changePredictionColNameChecker.doClick();
    }
    m_showExplanationChecker.setSelected(config.isShowExplanation());
    m_computeInteractionsChecker.setSelected(config.isComputeInteractions());
    removeAll();
    initLayout();
  }

  /**
   * Saves the settings to <b>settings</b>
   *
   * @param settings
   * @throws InvalidSettingsException
   */
  public void saveSettingsTo(final NodeSettingsWO settings) {
    final TreeSHAPConfiguration config = new TreeSHAPConfiguration(m_isRegression, "");
    config.setAppendClassConfidences(m_appendClassProbabilitiesColChecker.isSelected());
    config.setAppendPredictionConfidence(m_appendOverallConfidenceColChecker.isSelected());
    config.setPredictionColumnName(m_predictionColNameField.getText());
    config.setChangePredictionColumnName(m_changePredictionColNameChecker.isSelected());
    config.setSuffixForClassConfidences(m_suffixForClassProbabilitiesTextField.getText());
    config.setUseSoftVoting(m_useSoftVotingChecker.isSelected());
    config.setPositiveClass((DataCell) m_positiveClass.getSelectedItem());
    config.setShowExplanation(m_showExplanationChecker.isSelected());
    config.setComputeInteractions(m_computeInteractionsChecker.isSelected());
    config.save(settings);
  }
}
