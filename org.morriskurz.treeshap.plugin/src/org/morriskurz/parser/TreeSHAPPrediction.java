package org.morriskurz.parser;

import java.util.Arrays;

import org.knime.base.node.mine.treeensemble2.data.PredictorRecord;
import org.knime.base.node.mine.treeensemble2.model.AbstractTreeNode;
import org.knime.core.data.def.DoubleCell;

/**
 * Abstract implementation of the prediction interface calculation SHAP values. For details to the
 * implementation, refer to <a href=
 * "https://github.com/slundberg/shap/blob/master/shap/tree_shap.h">github.com/shap</a> For an
 * intuitive explanation on interaction values, see <a
 * href="https://link.springer.com/content/pdf/10.1007/s001820050125.pdf">here</a>
 *
 * @author Morris Kurz, morriskurz@gmail.com
 * @param <P> Whether the tree has regression or classification nodes. Note that GBTs are regression
 *     trees, one for each class.
 */
public abstract class TreeSHAPPrediction<P extends AbstractTreeNode>
    implements ITreeSHAPPrediction {

  private final PredictorRecord record;

  /** These are the SHAP values for the data row. The last value is the bias. */
  private double[] phi;

  /**
   * These are the interaction values for the data row. For a data row with n columns and a model
   * with p classes, (1 for binary), the interaction values are a (n+1)(n+1)p matrix. One matrix for
   * each class, with the SHAP values on the diagonal and the other SHAP values plus bias when a
   * feature is on/off.
   */
  private double[] interactionValues;

  private int nrColumns;

  protected TreeSHAPPrediction(final PredictorRecord record) {
    this.record = record;
  }

  /**
   * Calculates the SHAP values for the given record and tree roots. Uses the path_dependent
   * assumption. Corresponds to the tree_shap method.
   *
   * @param roots Roots of the trees making up the ensemble. Corresponds to trees
   * @param nrColumns The number of columns in the training data set. Corresponds to data.M
   * @param scaling The scaling factor which is applied to the SHAP values. Used to adhere with the
   *     c++ implementation.
   * @param maxDepth The maximum depth in a single tree over the whole ensemble. Corresponds to
   *     max_depth
   * @param condition The condition number in {-1, 0, 1}, corresponding to a feature being {off,
   *     normal, on}.
   * @param conditionFeature The feature index of the conditioned feature. Only relevant if
   *     conditon!=0.
   */
  public void computeShap(
      final P[] roots,
      final int nrColumns,
      final double scaling,
      final int maxDepth,
      final int condition,
      final int conditionFeature) {
    final int arraySize = (maxDepth + 2) * (maxDepth + 3) / 2;
    // Initializes to zero.
    this.phi = new double[nrColumns + 1];
    final PathElement[] uniquePath = new PathElement[arraySize];
    final double[] proportions = new double[arraySize];
    int index = 0;
    for (final P root : roots) {
      recurse(index, root, uniquePath, proportions, 0, 1, 1, -1, 0, condition, conditionFeature, 1);
      // Consider mean prediction. Bias of SHAP
      // TODO: Multi-output
      if (condition == 0) {
        phi[nrColumns] += getNodeValue(root, index);
      }
      index++;
    }
    for (int i = 0; i < nrColumns + 1; i++) {
      phi[i] *= scaling;
    }
  }

  /**
   * Calculates the SHAP interaction values for the given record and tree roots. Uses the
   * path_dependent assumption.
   *
   * @param roots Roots of the trees making up the ensemble
   * @param nrColumns The number of columns in the training data set.
   * @param scaling The scaling factor which is applied to the SHAP values. Used to adhere with the
   *     c++ implementation.
   * @param maxDepth The maximum depth in a single tree over the whole ensemble. Corresponds to
   *     max_depth
   * @param maxNodes The maximum amount of nodes in a single tree.
   * @param uniqueFeaturesPerTree A amountOfTrees*min(maxNodes, nrColumns) array containing the
   *     feature indeces used in each tree. Each tree has a min(maxNodes, nrColumns) space for its
   *     own unique features.
   */
  public void dense_tree_interactions_path_dependent(
      final P[] roots,
      final int nrColumns,
      final double scaling,
      final int maxDepth,
      final int maxNodes,
      final int[] uniqueFeaturesPerTree) {
    this.nrColumns = nrColumns;
    // build a list of all the unique features in each tree
    final int amountOfUniqueFeatures = Math.min(nrColumns, maxNodes);
    /*final int[] unique_features = new int[roots.length * maxNodes];
    for (int i = 0; i < unique_features.length; ++i) {
      unique_features[i] = -1;
    }
    // Iterate over every tree and filter every unique feature.
    // TODO: Do this before the prediction is called, so it is only done once.
    for (int j = 0; j < roots.length; ++j) {
        const final int[] features_row = nrColumns + j * maxNodes;
        final int *unique_features_row = unique_features + j * trees.max_nodes;
        for (unsigned k = 0; k < trees.max_nodes; ++k) {
            for (unsigned l = 0; l < trees.max_nodes; ++l) {
                if (features_row[k] == unique_features_row[l]) {
           break;
         }
                if (unique_features_row[l] < 0) {
                    unique_features_row[l] = features_row[k];
                    break;
                }
            }
        }
    }*/

    // build an interaction explanation for each sample
    // Usually one per class, but for now only binary classification
    // TODO: Multi-class
    // final int contrib_row_size = (nrColumns + 1) * trees.num_outputs;
    final int contrib_row_size = nrColumns + 1;
    final double[] diag_contribs = new double[contrib_row_size];
    final double[] on_contribs = new double[contrib_row_size];
    final double[] off_contribs = new double[contrib_row_size];
    interactionValues = new double[(nrColumns + 1) * contrib_row_size];
    // data.get_x_instance(instance, i); // record

    // aggregate the effect of explaining each tree
    // (this works because of the linearity property of Shapley values)
    // We compute the diagonal values beforehand since the implementation
    // of our computeShap handles an actual ensemble and not single trees.
    computeShap(roots, nrColumns, scaling, maxDepth, 0, 0);
    System.arraycopy(phi, 0, diag_contribs, 0, nrColumns + 1);
    P[] tree;
    for (int j = 0; j < roots.length; ++j) {
      // Since every tree needs to be considered seperately, we don't
      // just use computeShap on the whole array. Otherwise we would
      // do a lot of unneccessary computations for features that are
      // not present in some trees.
      tree = Arrays.copyOfRange(roots, j, j + 1);
      for (int k = 0; k < amountOfUniqueFeatures; ++k) {
        final int ind = uniqueFeaturesPerTree[k + j * amountOfUniqueFeatures];
        if (ind < 0) {
          // < 0 means we have seen all the features for this tree
          break;
        }

        // compute the shap value with this feature held on and off
        /*for (int index = 0; index < contrib_row_size; index++) {
          // Reused for every computation, so clear their values here.
          on_contribs[index] = 0;
          off_contribs[index] = 0;
        }*/
        computeShap(tree, nrColumns, scaling, maxDepth, 1, ind);
        System.arraycopy(phi, 0, on_contribs, 0, nrColumns + 1);
        computeShap(tree, nrColumns, scaling, maxDepth, -1, ind);
        System.arraycopy(phi, 0, off_contribs, 0, nrColumns + 1);

        // save the difference between on and off as the interaction value
        for (int l = 0; l < contrib_row_size; ++l) {
          final double val = (on_contribs[l] - off_contribs[l]) / 2;
          interactionValues[ind * contrib_row_size + l] += val;
          diag_contribs[l] -= val;
        }
      }
    }

    // set the diagonal
    for (int j = 0; j < nrColumns + 1; ++j) {
      // TODO: Multi-output
      final int offset = j * contrib_row_size + j * 1; // trees.num_outputs;
      for (int k = 0; k < 1 /*trees.num_outputs*/; ++k) {
        interactionValues[offset + k] = diag_contribs[j * 1 /*trees.num_outputs*/ + k];
      }
    }

    // apply the base offset to the bias term
    // TODO: Multi-output
    // final int last_ind = (nrColumns * (nrColumns + 1) + nrColumns) * 1;//trees.num_outputs;
    // for (int j = 0; j < 1/*trees.num_outputs*/ ++j) {
    //    interactionValues[last_ind + j] += trees.base_offset[j];
    // }

  }

  public void dense_tree_path_dependent(
      final P[] roots, final int nrColumns, final double scaling, final int maxDepth) {
    // TODO: Multi-output
    final double[] instance_out_contribs = new double[(nrColumns + 1) * 1 /* * trees.num_outputs*/];

    // aggregate the effect of explaining each tree
    // (this works because of the linearity property of Shapley values)
    computeShap(roots, nrColumns, scaling, maxDepth, 0, 0);

    // TODO: apply the base offset to the bias term.
    /*for (unsigned j = 0; j < trees.num_outputs; ++j) {
        instance_out_contribs[data.M * trees.num_outputs + j] += trees.base_offset[j];
    }*/

  }

  private void extendPath(
      final PathElement[] uniquePath,
      final double[] proportions,
      final int pathLength,
      final double zeroFraction,
      final double oneFraction,
      final int lastSplitFeatureIndex,
      final int currentPathIndex) {
    uniquePath[currentPathIndex + pathLength] =
        new PathElement(lastSplitFeatureIndex, zeroFraction, oneFraction);
    proportions[currentPathIndex + pathLength] = pathLength == 0 ? 1 : 0;
    for (int i = pathLength - 1; i >= 0; i--) {
      proportions[currentPathIndex + i + 1] +=
          oneFraction * proportions[currentPathIndex + i] * (i + 1) / (pathLength + 1);
      proportions[currentPathIndex + i] =
          zeroFraction * proportions[currentPathIndex + i] * (pathLength - i) / (pathLength + 1);
    }
  }

  @Override
  public DoubleCell[] getInteractionValues() {
    final DoubleCell[] newCells = new DoubleCell[nrColumns * nrColumns];
    /*for (int i = 0; i < interactionValues.length; i++) {
      // Skip every nrColumns+1 entry because that is the bias of the row
      newCells[i] = new DoubleCell(interactionValues[i]);
    }*/
    for (int row = 0; row < nrColumns; ++row) {
      for (int col = 0; col < nrColumns; ++col) {
        newCells[row * nrColumns + col] =
            new DoubleCell(interactionValues[row * (nrColumns + 1) + col]);
      }
    }
    return newCells;
  }

  /**
   * Mean prediction of this decision or regression tree node. For classification random forests,
   * this simply corresponds to the expected prediction for that decision tree. For regression
   * trees, this corresponds to the mean prediction.
   *
   * @param node The root node of the tree.
   * @param treeIndex Only for gradient boosted trees. Corresponds to the index of the tree we want
   *     to access.
   */
  protected abstract double getNodeValue(P node, int treeIndex);

  /** Returns the number of samples going through that tree node. */
  protected abstract double getNumberOfSamples(P node);

  public double[] getPhi() {
    return phi;
  }

  @Override
  public DoubleCell[] getSHAPValues() {
    final DoubleCell[] newCells = new DoubleCell[phi.length];
    for (int i = 0; i < phi.length; i++) {
      newCells[i] = new DoubleCell(phi[i]);
    }
    return newCells;
  }

  private boolean isLeaf(final P node) {
    return node.getNrChildren() == 0;
  }

  /**
   * Recurse on the root node of a decision tree to calculate the SHAP values.
   *
   * @param currentNode The current node which is operated on.
   * @param uniqueParentPath Array of all encountered PathElements in this tree branch. It should
   *     have a size of at least (maxDepth + 2)*(maxDepth+3).
   * @param parentProprotions Array of the proportions of the path elements. It should have a size
   *     of at least (maxDepth + 2)*(maxDepth+3).
   * @param pathLength Current path length. The path length equals the depth of the current node.
   * @param zeroFraction Fraction of "zero" paths passing through this node.
   * @param oneFraction Fraction of "one" paths passing through this node.
   * @param lastSplitFeatureIndex The feature index the last split was performed on. For the root
   *     node, this should be -1.
   * @param parentPathIndex The array start index for the parent's path element. The range
   *     [parentPathIndex, ..., parentPathIndex + pathLength] (inclusive) belongs to the parent.
   * @param condition Either 0, -1 or 1. 0 is the normal treeshap algorithm, while -1 and 1
   *     correspond to a feature being off or on.
   * @param conditionFeature Index of the feature being conditioned on. Only relevant if condition
   *     != 0.
   * @param conditionFraction The incoming conditionFraction.
   */
  private void recurse(
      final int treeIndex,
      final P currentNode,
      final PathElement[] uniqueParentPath,
      final double[] parentProprotions,
      int pathLength,
      final double zeroFraction,
      final double oneFraction,
      final int lastSplitFeatureIndex,
      final int parentPathIndex,
      final int condition,
      final int conditionFeature,
      final double conditionFraction) {
    // stop if we have no weight coming down to us
    if (conditionFraction == 0) {
      return;
    }

    System.arraycopy(
        uniqueParentPath,
        parentPathIndex,
        uniqueParentPath,
        parentPathIndex + pathLength + 1,
        pathLength + 1);
    System.arraycopy(
        parentProprotions,
        parentPathIndex,
        parentProprotions,
        parentPathIndex + pathLength + 1,
        pathLength + 1);
    final int currentPathIndex = parentPathIndex + pathLength + 1;

    if (condition == 0 || conditionFeature != lastSplitFeatureIndex) {
      extendPath(
          uniqueParentPath,
          parentProprotions,
          pathLength,
          zeroFraction,
          oneFraction,
          lastSplitFeatureIndex,
          currentPathIndex);
    }
    if (isLeaf(currentNode)) {
      for (int i = 1; i <= pathLength; ++i) {
        final double w =
            unwoundPathSum(uniqueParentPath, parentProprotions, pathLength, i, currentPathIndex);
        final PathElement el = uniqueParentPath[currentPathIndex + i];
        final double scale =
            w * (el.getFractionOfOnePaths() - el.getFractionOfZeroPaths()) * conditionFraction;
        // SHAP Value for the positive class.
        if (el.getFeatureIndexForSplit() == -1) {
          System.out.println(
              "ERROR on recurse: "
                  + i
                  + ", "
                  + pathLength
                  + ", "
                  + lastSplitFeatureIndex
                  + ", "
                  + conditionFeature);
        } else {
          phi[el.getFeatureIndexForSplit()] += scale * getNodeValue(currentNode, treeIndex);
        }
      }
    } else {
      final int splitIndex = currentNode.getSplitAttributeIndex();
      if (splitIndex == -1) {
        System.out.println("ERROR on splitindex: " + currentNode.toString() + ", ");
      }
      // find which branch is "hot" (meaning the instance would follow it)
      int hotIndex = currentNode.findNextPathTurn(record);
      if (hotIndex == -1) {
        // Missing value. Default child is the left one.
        hotIndex = 0;
      }
      // TODO: What if the tree is not binary?
      if (currentNode.getNrChildren() > 2) {
        throw new IllegalArgumentException(
            "A node has more than two children nodes in your tree! Make sure you don't have a tree with non-binary splits. "
                + currentNode.toString());
      }
      final int coldIndex = (1 - hotIndex) % 2;
      @SuppressWarnings("unchecked")
      final P hotChild = (P) currentNode.getChild(hotIndex);
      @SuppressWarnings("unchecked")
      final P coldChild = (P) currentNode.getChild(coldIndex);
      final double w = getNumberOfSamples(currentNode);
      final double hotZeroFraction = getNumberOfSamples(hotChild) / w;
      final double coldZeroFraction = getNumberOfSamples(coldChild) / w;
      double incomingZeroFraction = 1;
      double incomingOneFraction = 1;

      // see if we have already split on this feature,
      // if so we undo that split so we can redo it for this node
      int pathIndex = 0;
      for (; pathIndex <= pathLength; ++pathIndex) {
        if (uniqueParentPath[currentPathIndex + pathIndex] == null) {
          continue;
        }
        if (uniqueParentPath[currentPathIndex + pathIndex].getFeatureIndexForSplit()
            == splitIndex) {
          break;
        }
      }
      if (pathIndex != pathLength + 1) {
        incomingZeroFraction =
            uniqueParentPath[currentPathIndex + pathIndex].getFractionOfZeroPaths();
        incomingOneFraction =
            uniqueParentPath[currentPathIndex + pathIndex].getFractionOfOnePaths();
        unwindPath(uniqueParentPath, parentProprotions, pathLength, pathIndex, currentPathIndex);
        pathLength -= 1;
      }

      // divide up the condition_fraction among the recursive calls
      double hotConditionFraction = conditionFraction;
      double coldConditionFraction = conditionFraction;
      if (condition > 0 && splitIndex == conditionFeature) {
        coldConditionFraction = 0;
        pathLength -= 1;
      } else if (condition < 0 && splitIndex == conditionFeature) {
        hotConditionFraction *= hotZeroFraction;
        coldConditionFraction *= coldZeroFraction;
        pathLength -= 1;
      }

      recurse(
          treeIndex,
          hotChild,
          uniqueParentPath,
          parentProprotions,
          pathLength + 1,
          hotZeroFraction * incomingZeroFraction,
          incomingOneFraction,
          splitIndex,
          currentPathIndex,
          condition,
          conditionFeature,
          hotConditionFraction);
      recurse(
          treeIndex,
          coldChild,
          uniqueParentPath,
          parentProprotions,
          pathLength + 1,
          coldZeroFraction * incomingZeroFraction,
          0,
          splitIndex,
          currentPathIndex,
          condition,
          conditionFeature,
          coldConditionFraction);
    }
  }

  private void unwindPath(
      final PathElement[] uniquePath,
      final double[] proportions,
      final int pathLength,
      final int pathIndex,
      final int currentPathIndex) {
    final double one_fraction = uniquePath[pathIndex + currentPathIndex].getFractionOfOnePaths();
    final double zero_fraction = uniquePath[pathIndex + currentPathIndex].getFractionOfZeroPaths();
    double nextOneProportion = proportions[pathLength + currentPathIndex];

    for (int i = pathLength - 1; i >= 0; --i) {
      if (one_fraction != 0) {
        final double tmp = proportions[i + currentPathIndex];
        proportions[i + currentPathIndex] =
            nextOneProportion * (pathLength + 1) / ((i + 1) * one_fraction);
        nextOneProportion =
            tmp
                - proportions[i + currentPathIndex]
                    * zero_fraction
                    * (pathLength - i)
                    / (pathLength + 1);
      } else {
        proportions[i + currentPathIndex] =
            proportions[i + currentPathIndex]
                * (pathLength + 1)
                / (zero_fraction * (pathLength - i));
      }
    }

    for (int i = pathIndex; i < pathLength; ++i) {
      uniquePath[i + currentPathIndex] =
          new PathElement(
              uniquePath[currentPathIndex + i + 1].getFeatureIndexForSplit(),
              uniquePath[currentPathIndex + i + 1].getFractionOfZeroPaths(),
              uniquePath[currentPathIndex + i + 1].getFractionOfOnePaths());
    }
  }

  private double unwoundPathSum(
      final PathElement[] uniquePath,
      final double[] proportions,
      final int pathLength,
      final int index,
      final int currentPathIndex) {
    final double oneFraction = uniquePath[currentPathIndex + index].getFractionOfOnePaths();
    final double zeroFraction = uniquePath[currentPathIndex + index].getFractionOfZeroPaths();
    double nextOnePortion = proportions[currentPathIndex + pathLength];
    double total = 0;

    if (oneFraction != 0) {
      for (int i = pathLength - 1; i >= 0; --i) {
        final double tmp = nextOnePortion / ((i + 1) * oneFraction);
        total += tmp;
        nextOnePortion = proportions[currentPathIndex + i] - tmp * zeroFraction * (pathLength - i);
      }
    } else {
      for (int i = pathLength - 1; i >= 0; --i) {
        total += proportions[currentPathIndex + i] / (zeroFraction * (pathLength - i));
      }
    }
    return total * (pathLength + 1);
  }
}
