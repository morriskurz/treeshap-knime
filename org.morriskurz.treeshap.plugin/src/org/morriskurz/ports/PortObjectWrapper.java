package org.morriskurz.ports;

import org.knime.core.data.DataTableSpec;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;

public class PortObjectWrapper<T extends PortObject, S extends PortObjectSpec> extends Ports {

	private final PortObject[] portObjects;

	private final PortSpecWrapper<S> portSpecs;

	public PortObjectWrapper(final PortObject... portObjects) throws InvalidSettingsException {
		this.portObjects = portObjects;
		final PortObjectSpec[] specs = new PortObjectSpec[portObjects.length];
		int i = 0;
		for (final PortObject portObject : portObjects) {
			if (portObject == null) {
				specs[i] = null;
			} else {
				specs[i] = portObject.getSpec();
			}
			i++;
		}
		portSpecs = new PortSpecWrapper<>(specs);
	}

	public BufferedDataTable getData() {
		return (BufferedDataTable) portObjects[IN_DATA_PORT_INDEX];
	}

	public DataTableSpec getDataSpec() {
		return portSpecs.getDataSpec();
	}

	@SuppressWarnings("unchecked")
	public T getModel() {
		return (T) portObjects[IN_MODEL_PORT_INDEX];
	}

	public S getModelSpec() {
		return portSpecs.getModelSpec();
	}

}
