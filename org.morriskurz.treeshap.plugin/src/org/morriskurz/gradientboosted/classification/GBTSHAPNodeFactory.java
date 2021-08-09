package org.morriskurz.gradientboosted.classification;

import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeFactory;
import org.knime.core.node.NodeView;
import org.morriskurz.dialog.TreeSHAPNodeDialog;

/**
 * This is an example implementation of the node factory of the "TreeSHAP" node.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public class GBTSHAPNodeFactory extends NodeFactory<GBTSHAPNodeModel> {

  /** {@inheritDoc} */
  @Override
  public NodeDialogPane createNodeDialogPane() {
    // This example node has a dialog, hence we create and return it here. Also see
    // "hasDialog()".
    return new TreeSHAPNodeDialog(false, false);
  }

  /** {@inheritDoc} */
  @Override
  public GBTSHAPNodeModel createNodeModel() {
    // Create and return a new node model.
    return new GBTSHAPNodeModel();
  }

  /** {@inheritDoc} */
  @Override
  public NodeView<GBTSHAPNodeModel> createNodeView(
      final int viewIndex, final GBTSHAPNodeModel nodeModel) {
    // We return null as this example node does not provide a view. Also see
    // "getNrNodeViews()".
    throw new IndexOutOfBoundsException();
  }

  /** {@inheritDoc} */
  @Override
  public int getNrNodeViews() {
    // The number of views the node should have, in this cases there is none.
    return 0;
  }

  /** {@inheritDoc} */
  @Override
  public boolean hasDialog() {
    // Indication whether the node has a dialog or not.
    return true;
  }
}
