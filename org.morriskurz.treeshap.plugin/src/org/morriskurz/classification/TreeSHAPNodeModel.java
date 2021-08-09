package org.morriskurz.classification;

import java.io.File;
import java.io.IOException;
import java.util.Optional;

import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModelPortObject;
import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModelPortObjectSpec;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.container.ColumnRearranger;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.streamable.InputPortRole;
import org.knime.core.node.streamable.OutputPortRole;
import org.knime.core.node.streamable.PartitionInfo;
import org.knime.core.node.streamable.PortInput;
import org.knime.core.node.streamable.PortObjectInput;
import org.knime.core.node.streamable.PortOutput;
import org.knime.core.node.streamable.StreamableFunction;
import org.knime.core.node.streamable.StreamableOperator;
import org.morriskurz.TreeSHAPConfiguration;
import org.morriskurz.TreeSHAPUtil;
import org.morriskurz.ports.PortObjectWrapper;
import org.morriskurz.ports.PortSpecWrapper;
import org.morriskurz.ports.Ports;

/**
 * This is the implementation of the node model of the "TreeSHAP" node.
 *
 * <p>It calculates the SHAP values in polynomial time. It is based on Lundberg, Erion and Lee
 * (2018) TreeSHAP algorithm.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public class TreeSHAPNodeModel extends NodeModel {
  /**
   * The logger is used to print info/warning/error messages to the KNIME console and to the KNIME
   * log file. Retrieve it via 'NodeLogger.getLogger' providing the class of this node model.
   */
  private static final NodeLogger LOGGER = NodeLogger.getLogger(TreeSHAPNodeModel.class);

  private TreeSHAPConfiguration configuration;

  /** Constructor for the node model. */
  protected TreeSHAPNodeModel() {
    /** Here we specify how many data input and output tables the node should have. */
    super(Ports.INPUT_PORTS, Ports.OUTPUT_PORTS);
  }

  /** {@inheritDoc} */
  @Override
  protected PortObjectSpec[] configure(final PortObjectSpec[] inSpecs)
      throws InvalidSettingsException {
    /*
     * Check if the node is executable, e.g. all required user settings are
     * available and valid, or the incoming types are feasible for the node to
     * execute. In case the node can execute in its current configuration with the
     * current input, calculate and return the table spec that would result of the
     * execution of this node. I.e. this method precalculates the table spec of the
     * output table.
     */
    final PortSpecWrapper<TreeEnsembleModelPortObjectSpec> specs = new PortSpecWrapper<>(inSpecs);
    specs.getModelSpec().assertTargetTypeMatches(false);
    /*
     * Similar to the return type of the execute method, we need to return an array
     * of DataTableSpecs with the length of the number of outputs ports of the node
     * (as specified in the constructor). The resulting table created in the execute
     * methods must match the spec created in this method.
     */
    if (configuration == null) {
      configuration =
          TreeSHAPConfiguration.createDefault(
              false, specs.getModelSpec().getTargetColumn().getName());
    }
    if (configuration.getPositiveClass() == null) {
      configuration.setPositiveClass(
          specs.getModelSpec().getTargetColumnPossibleValueMap().values().iterator().next());
      setWarningMessage(
          "No value for the positive class selected, had to guess "
              + configuration.getPositiveClass().toString()
              + " as positive class.");
    }
    final Optional<DataTableSpec> outSpecOptional = createOutputSpec(specs);
    if (outSpecOptional.isPresent()) {
      return new PortObjectSpec[] {outSpecOptional.get()};
    }
    return new PortObjectSpec[] {};
  }

  /**
   * Creates the output table spec from the input spec. For each column used in the prediction by
   * the tree ensemble, one String column will be created containing the corresponding SHAP values.
   *
   * @param inputTableSpec
   * @return outputTableSpec
   */
  private Optional<DataTableSpec> createOutputSpec(
      final PortSpecWrapper<TreeEnsembleModelPortObjectSpec> specs)
      throws InvalidSettingsException {
    final TreeEnsembleModelPortObjectSpec modelSpec = specs.getModelSpec();
    final DataTableSpec dataSpec = specs.getDataSpec();
    Optional<DataTableSpec> outSpec;
    outSpec =
        TreeSHAPUtil.createPRCForClassificationRF(dataSpec, modelSpec, null, configuration)
            .createSpec();
    return outSpec;
  }

  /** {@inheritDoc} */
  @Override
  public StreamableOperator createStreamableOperator(
      final PartitionInfo partitionInfo, final PortObjectSpec[] inSpecs)
      throws InvalidSettingsException {
    return new StreamableOperator() {

      @Override
      public void runFinal(
          final PortInput[] inputs, final PortOutput[] outputs, final ExecutionContext exec)
          throws Exception {
        final TreeEnsembleModelPortObject model =
            (TreeEnsembleModelPortObject) ((PortObjectInput) inputs[0]).getPortObject();
        final TreeEnsembleModelPortObjectSpec modelSpec = model.getSpec();
        final DataTableSpec dataSpec = (DataTableSpec) inSpecs[1];
        final ColumnRearranger rearranger =
            TreeSHAPUtil.createPRCForClassificationRF(
                    dataSpec, modelSpec, model.getEnsembleModel(), configuration)
                .createExecutionRearranger();
        final StreamableFunction func = rearranger.createStreamableFunction(1, 0);
        func.runFinal(inputs, outputs, exec);
      }
    };
  }

  /** {@inheritDoc} */
  @Override
  protected PortObject[] execute(final PortObject[] inObjects, final ExecutionContext exec)
      throws Exception {
    final PortObjectWrapper<TreeEnsembleModelPortObject, TreeEnsembleModelPortObjectSpec>
        portObjects = new PortObjectWrapper<>(inObjects);
    final TreeEnsembleModelPortObject model = portObjects.getModel();
    final TreeEnsembleModelPortObjectSpec modelSpec = model.getSpec();
    final BufferedDataTable inData = portObjects.getData();
    final DataTableSpec dataSpec = inData.getDataTableSpec();
    if (!portObjects.getModel().getEnsembleModel().containsClassDistribution()) {
      setWarningMessage(
          "Activate the \"Save target distribution in tree nodes\" checkbox in the learner node to access the true SHAP values! They are not accurate without this option enabled.");
    }

    ColumnRearranger rearranger;
    configuration.checkSoftVotingSettingForModel(model).ifPresent(this::setWarningMessage);
    rearranger =
        TreeSHAPUtil.createPRCForClassificationRF(
                dataSpec, modelSpec, model.getEnsembleModel(), configuration)
            .createExecutionRearranger();
    LOGGER.debug("Rearranger was succesfully built.");
    final BufferedDataTable outTable = exec.createColumnRearrangeTable(inData, rearranger, exec);
    return new BufferedDataTable[] {outTable};
  }

  /** {@inheritDoc} */
  @Override
  public InputPortRole[] getInputPortRoles() {
    return new InputPortRole[] {
      InputPortRole.NONDISTRIBUTED_NONSTREAMABLE, InputPortRole.DISTRIBUTED_STREAMABLE
    };
  }

  /** {@inheritDoc} */
  @Override
  public OutputPortRole[] getOutputPortRoles() {
    return new OutputPortRole[] {OutputPortRole.DISTRIBUTED};
  }

  @Override
  protected void loadInternals(final File nodeInternDir, final ExecutionMonitor exec)
      throws IOException, CanceledExecutionException { // No internals
  }

  /** {@inheritDoc} */
  @Override
  protected void loadValidatedSettingsFrom(final NodeSettingsRO settings)
      throws InvalidSettingsException {
    /*
     * Load (valid) settings from the NodeSettings object. It can be safely assumed
     * that the settings are validated by the method below.
     */
    final TreeSHAPConfiguration config = new TreeSHAPConfiguration(false, "");
    config.loadInModel(settings);
    configuration = config;
  }

  @Override
  protected void reset() { // No internals
  }

  @Override
  protected void saveInternals(final File nodeInternDir, final ExecutionMonitor exec)
      throws IOException, CanceledExecutionException { // No internals
  }

  /** {@inheritDoc} */
  @Override
  protected void saveSettingsTo(final NodeSettingsWO settings) {
    /*
     * Save user settings to the NodeSettings object. SettingsModels already know
     * how to save them self to a NodeSettings object by calling the below method.
     * In general, the NodeSettings object is just a key-value store and has methods
     * to write all common data types. Hence, you can easily write your settings
     * manually. See the methods of the NodeSettingsWO.
     */
    if (configuration != null) {
      configuration.save(settings);
    }
  }

  /** {@inheritDoc} */
  @Override
  protected void validateSettings(final NodeSettingsRO settings) throws InvalidSettingsException {
    /*
     * Check if the settings could be applied to our model e.g. if the user provided
     * format String is empty. In this case we do not need to check as this is
     * already handled in the dialog. Do not actually set any values of any member
     * variables.
     */
    final TreeSHAPConfiguration config = new TreeSHAPConfiguration(false, "");
    config.loadInModel(settings);
  }
}
