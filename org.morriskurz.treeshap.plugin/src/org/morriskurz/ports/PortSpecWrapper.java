package org.morriskurz.ports;

import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModelPortObjectSpec;
import org.knime.core.data.DataTableSpec;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.port.PortObjectSpec;

/**
 * Wrapper object for the port object specs.
 * 
 * @author Morris Kurz, morriskurz@gmail.com
 *
 * @param <T> The model specs, i.e. TreeEnsembleSpec or GBTSpec
 */
public class PortSpecWrapper<T extends PortObjectSpec> extends Ports {

	private final PortObjectSpec[] specs;

	public PortSpecWrapper(final PortObjectSpec... specs) throws InvalidSettingsException {
		this.specs = specs;
		sanityCheckInput(specs);
	}

	public DataTableSpec getDataSpec() {
		return (DataTableSpec) specs[IN_DATA_PORT_INDEX];
	}

	@SuppressWarnings("unchecked")
	public T getModelSpec() {
		return (T) specs[IN_MODEL_PORT_INDEX];
	}

	private void sanityCheckInput(final PortObjectSpec... specs) throws InvalidSettingsException {
		if (specs.length != 2 || !(specs[IN_DATA_PORT_INDEX] instanceof DataTableSpec)
				|| !(specs[IN_MODEL_PORT_INDEX] instanceof TreeEnsembleModelPortObjectSpec)) {
			throw new InvalidSettingsException(
					"The specs are not correct. Did you assign the " + "correct port roles in the NodeModel?");
		}
	}

}
