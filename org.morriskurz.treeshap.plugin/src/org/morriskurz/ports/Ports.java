package org.morriskurz.ports;

import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModelPortObject;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.port.PortType;

/**
 * Data class for the port objects. Defines the input and output ports, as well
 * as the corresponding indexes.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 *
 */
public class Ports {

	public static final PortType[] INPUT_PORTS = { TreeEnsembleModelPortObject.TYPE, BufferedDataTable.TYPE };
	public static final PortType[] OUTPUT_PORTS = { BufferedDataTable.TYPE };

	protected static final int IN_MODEL_PORT_INDEX = 0;
	protected static final int IN_DATA_PORT_INDEX = 1;

	protected Ports() {

	}
}
