package org.morriskurz.statistics;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.knime.base.node.mine.treeensemble2.model.AbstractTreeNode;
import org.knime.base.node.mine.treeensemble2.model.MultiClassGradientBoostedTreesModel;
import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModel;

/**
 * Data object holding information about a tree ensemble.
 *
 * <p>Saves the maximum depth, maximum amount of nodes in a tree and the unique features per tree.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public class TreeEnsembleStatistics {

  private int maxNodes;

  private int maxDepth;

  private final ArrayList<Set<Integer>> uniqueFeaturesPerTreeList;

  private List<Map<AbstractTreeNode, Double>> gradientBoostedTreesMaps;

  public TreeEnsembleStatistics() {
    uniqueFeaturesPerTreeList = new ArrayList<>();
  }

  public int getMaximumDepth() {
    return maxDepth;
  }

  public int getMaximumNumberOfNodes() {
    return maxNodes;
  }

  /**
   * @param nrModels
   * @param amountOfUniqueFeatures = min(maxNodes, nrFeatures)
   * @return
   */
  public int[] getUniqueFeaturesPerTree(final int nrModels, final int amountOfUniqueFeatures) {
    final int[] uniqueFeaturesPerTree = new int[nrModels * amountOfUniqueFeatures];
    for (int i = 0; i < nrModels * amountOfUniqueFeatures; i++) {
      // Important to initialize such that the algorithm knows when there
      // are no unique features left.
      uniqueFeaturesPerTree[i] = -1;
    }
    for (int i = 0; i < nrModels; i++) {
      final Set<Integer> treeFeatures = uniqueFeaturesPerTreeList.get(i);
      int currentIndex = i * amountOfUniqueFeatures;
      for (final Integer splitIndex : treeFeatures) {
        uniqueFeaturesPerTree[currentIndex] = splitIndex;
        currentIndex++;
      }
    }
    return uniqueFeaturesPerTree;
  }

  public void initialize(final MultiClassGradientBoostedTreesModel forest) {
    for (int i = 0; i < forest.getNrModels(); i++) {
      uniqueFeaturesPerTreeList.add(new HashSet<>());
      final AbstractTreeNode root = forest.getTreeModel(i).getRootNode();
      final ArrayList<AbstractTreeNode> currentNodeList = new ArrayList<>();
      final ArrayList<AbstractTreeNode> nextNodeList = new ArrayList<>();
      currentNodeList.add(root);
      int depth = 0;
      int amountOfNodes = 0;
      // Breadth-first traversal of the tree
      while (!currentNodeList.isEmpty()) {
        depth++;
        for (final AbstractTreeNode node : currentNodeList) {
          amountOfNodes++;
          if (node.getNrChildren() > 0) {
            nextNodeList.addAll(node.getChildren());
          }
          final int featureSplitIndex = node.getSplitAttributeIndex();
          if (featureSplitIndex != -1) {
            uniqueFeaturesPerTreeList.get(i).add(featureSplitIndex);
          }
        }
        currentNodeList.clear();
        currentNodeList.addAll(nextNodeList);
        nextNodeList.clear();
      }
      maxDepth = Math.max(maxDepth, depth);
      maxNodes = Math.max(maxNodes, amountOfNodes);
    }
  }

  public void initialize(final TreeEnsembleModel forest) {
    for (int i = 0; i < forest.getNrModels(); i++) {
      uniqueFeaturesPerTreeList.add(new HashSet<>());
      final AbstractTreeNode root = forest.getTreeModel(i).getRootNode();
      final ArrayList<AbstractTreeNode> currentNodeList = new ArrayList<>();
      final ArrayList<AbstractTreeNode> nextNodeList = new ArrayList<>();
      currentNodeList.add(root);
      int depth = 0;
      int amountOfNodes = 0;
      // Breadth-first traversal of the tree
      while (!currentNodeList.isEmpty()) {
        depth++;
        for (final AbstractTreeNode node : currentNodeList) {
          amountOfNodes++;
          if (node.getNrChildren() > 0) {
            nextNodeList.addAll(node.getChildren());
          }
          final int featureSplitIndex = node.getSplitAttributeIndex();
          if (featureSplitIndex != -1) {
            uniqueFeaturesPerTreeList.get(i).add(featureSplitIndex);
          }
        }
        currentNodeList.clear();
        currentNodeList.addAll(nextNodeList);
        nextNodeList.clear();
      }
      maxDepth = Math.max(maxDepth, depth);
      maxNodes = Math.max(maxNodes, amountOfNodes);
    }
  }
}
