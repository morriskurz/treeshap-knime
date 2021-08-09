package org.morriskurz;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

import org.knime.base.data.filter.column.FilterColumnRow;
import org.knime.base.node.mine.treeensemble2.data.PredictorRecord;
import org.knime.base.node.mine.treeensemble2.model.AbstractTreeEnsembleModel;
import org.knime.base.node.mine.treeensemble2.model.AbstractTreeNode;
import org.knime.base.node.mine.treeensemble2.model.GradientBoostedTreesModel;
import org.knime.base.node.mine.treeensemble2.model.MultiClassGradientBoostedTreesModel;
import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModel;
import org.knime.base.node.mine.treeensemble2.model.TreeEnsembleModelPortObjectSpec;
import org.knime.base.node.mine.treeensemble2.model.TreeNodeRegression;
import org.knime.base.node.mine.treeensemble2.model.TreeNodeSignature;
import org.knime.base.node.mine.treeensemble2.node.predictor.PredictionRearrangerCreator;
import org.knime.base.node.mine.treeensemble2.node.predictor.TreeEnsemblePredictorConfiguration;
import org.knime.base.node.mine.treeensemble2.node.predictor.classification.HardVotingFactory;
import org.knime.base.node.mine.treeensemble2.node.predictor.classification.SoftVotingFactory;
import org.knime.base.node.mine.treeensemble2.node.predictor.classification.VotingFactory;
import org.knime.core.data.DataCell;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.container.ColumnRearranger;
import org.knime.core.node.InvalidSettingsException;
import org.morriskurz.classification.ExplanationClassificationPredictor;
import org.morriskurz.gradientboosted.classification.ExplanationGBTPredictor;
import org.morriskurz.gradientboosted.regression.ExplanationGBTRegressionPredictor;
import org.morriskurz.parser.ExplanationItemParser;
import org.morriskurz.parser.InteractionItemParser;
import org.morriskurz.regression.ExplanationRegressionPredictor;
import org.morriskurz.statistics.TreeEnsembleStatistics;

/**
 * Utility methods for creating PredictionRearrangerCreators for GBT and RF models.
 *
 * @author Morris Kurz, morriskurz@gmail.com
 */
public class TreeSHAPUtil {
  public static class Pair<T, U> {
    public final T first;
    public final U second;

    public Pair(final T t, final U u) {
      this.first = t;
      this.second = u;
    }
  }

  private static final String CONFIDENCE_SUFFIX = " (Confidence)";

  private static void addClassProbabilites(
      final PredictionRearrangerCreator crc,
      final String prefix,
      final TreeEnsembleModelPortObjectSpec modelSpec,
      final MultiClassGradientBoostedTreesModel model,
      final TreeEnsemblePredictorConfiguration config)
      throws InvalidSettingsException {
    crc.addClassProbabilities(
        modelSpec.getTargetColumnPossibleValueMap(),
        model == null ? null : model.getClassLabels(),
        prefix,
        config.getSuffixForClassProbabilities(),
        modelSpec.getTargetColumn().getName());
  }

  private static Pair<Integer, List<Map<AbstractTreeNode, Double>>> computeExpectations(
      final TreeNodeRegression[] trees,
      final List<Map<TreeNodeSignature, Double>> coefficientMaps) {
    final int nrModels = trees.length;
    int maxDepth = 0;
    final ArrayList<Map<AbstractTreeNode, Double>> treeMaps = new ArrayList<>(nrModels);
    for (int treeNumber = 0; treeNumber < nrModels; treeNumber++) {
      final Map<AbstractTreeNode, Double> expectations = new HashMap<>();
      final int depth =
          descendRegressionTree(
              coefficientMaps.get(treeNumber), trees[treeNumber], 0, expectations);

      treeMaps.add(expectations);
      maxDepth = Math.max(maxDepth, depth);
    }
    final Pair<Integer, List<Map<AbstractTreeNode, Double>>> p = new Pair<>(maxDepth, treeMaps);
    return p;
  }

  /**
   * Recomputes the expectations values as prescribed by
   * https://github.com/slundberg/shap/blob/master/shap/tree_shap.h#L492 The mean value in GBT is
   * not the correct prediction. The correct prediction is only stored for the leaves in the
   * CoefficientMap. The maximum depth is calculated additionally.
   *
   * @param model
   * @param positiveClassIndex
   * @return
   */
  private static Pair<Integer, List<Map<AbstractTreeNode, Double>>>
      computeExpectationsClassificationGBT(
          final MultiClassGradientBoostedTreesModel model, final int positiveClassIndex) {
    final int nrModels = model.getNrLevels();
    final TreeNodeRegression[] roots = new TreeNodeRegression[nrModels];
    final List<Map<TreeNodeSignature, Double>> coefficientMaps = new ArrayList<>(nrModels);
    for (int treeNumber = 0; treeNumber < nrModels; treeNumber++) {
      // Each tree predicting the positive class is extracted.
      // TODO: Multi-class
      roots[treeNumber] = model.getModel(treeNumber, positiveClassIndex).getRootNode();
      coefficientMaps.add(model.getCoefficientMap(treeNumber, positiveClassIndex));
    }
    return computeExpectations(roots, coefficientMaps);
  }

  private static Pair<Integer, List<Map<AbstractTreeNode, Double>>>
      computeExpectationsRegressionGBT(final GradientBoostedTreesModel model) {
    final TreeNodeRegression[] roots = new TreeNodeRegression[model.getNrModels()];
    for (int treeNumber = 0; treeNumber < model.getNrModels(); treeNumber++) {
      roots[treeNumber] = model.getTreeModelRegression(treeNumber).getRootNode();
    }
    return computeExpectations(roots, new ArrayList<>(model.getCoeffientMaps()));
  }

  /**
   * Creates a {@link PredictionRearrangerCreator} for creation of a {@link ColumnRearranger} that
   * can be used to predict with a classification random forest.
   *
   * @param dataSpec the spec of the table to predict
   * @param modelSpec the spec of the (classification) random forest
   * @param model the (classification) random forest
   * @param modelRowSamples row samples used to train the individual trees (may be null)
   * @param targetColumnData the target column (may be null)
   * @param config for the prediction
   * @return a creator that allows to create a rearranger for prediction with a random forest
   * @throws InvalidSettingsException if <b>dataSpec</b> is missing some columns the model needs
   */
  public static PredictionRearrangerCreator createPRCForClassificationRF(
      final DataTableSpec dataSpec,
      final TreeEnsembleModelPortObjectSpec modelSpec,
      final TreeEnsembleModel model,
      final TreeSHAPConfiguration config)
      throws InvalidSettingsException {
    final Map<String, DataCell> targetValueMap = modelSpec.getTargetColumnPossibleValueMap();
    ExplanationClassificationPredictor predictor = null;
    String[] classLabels = null;
    if (targetValueMap != null) {
      final Map<String, Integer> targetVal2Idx = createTargetValueToIndexMap(targetValueMap);
      final VotingFactory votingFactory =
          config.isUseSoftVoting()
              ? new SoftVotingFactory(targetVal2Idx)
              : new HardVotingFactory(targetVal2Idx);
      int maxDepth = -1;
      int maxNodes = 0;
      int[] uniqueFeaturesPerTree = null;
      final DataCell positiveClassCell = config.getPositiveClass();
      final String positiveClassName = positiveClassCell.toString();
      final int positiveClassIndex =
          positiveClassName == null ? 0 : targetVal2Idx.get(positiveClassName);
      if (model != null && config.isShowExplanation()) {
        final TreeEnsembleStatistics s = new TreeEnsembleStatistics();
        s.initialize(model);
        maxDepth = s.getMaximumDepth();
        if (config.isComputeInteractions()) {
          maxNodes = s.getMaximumNumberOfNodes();
          uniqueFeaturesPerTree =
              s.getUniqueFeaturesPerTree(
                  model.getNrModels(),
                  Math.min(maxNodes, modelSpec.getTableSpec().getNumColumns() - 1));
          // maxDepth = getMaxDepth(model);
        }
      }
      predictor =
          new ExplanationClassificationPredictor(
              model,
              modelSpec,
              dataSpec,
              votingFactory,
              maxDepth,
              positiveClassIndex,
              maxNodes,
              uniqueFeaturesPerTree);
      classLabels =
          targetValueMap
              .keySet()
              .stream()
              .map(
                  new Function<String, String>() {
                    @Override
                    public String apply(final String o) {
                      return o;
                    }
                  })
              .toArray(
                  new IntFunction<String[]>() {
                    @Override
                    public String[] apply(final int i) {
                      return new String[i];
                    }
                  });
    }

    final PredictionRearrangerCreator prc = new PredictionRearrangerCreator(dataSpec, predictor);

    if (config.isAppendClassConfidences()) {
      prc.addClassProbabilities(
          targetValueMap,
          classLabels,
          "P (",
          config.getSuffixForClassProbabilities(),
          modelSpec.getTargetColumn().getName());
    }
    prc.addClassPrediction(config.getPredictionColumnName());
    if (config.isAppendPredictionConfidence()) {
      prc.addPredictionConfidence(config.getPredictionColumnName() + CONFIDENCE_SUFFIX);
    }
    if (config.isShowExplanation()) {
      if (config.isComputeInteractions()) {
        prc.addPredictionItemParser(new InteractionItemParser(modelSpec.getTableSpec()));
      } else {
        prc.addPredictionItemParser(new ExplanationItemParser(modelSpec.getTableSpec()));
      }
    }
    if (config.isAppendModelCount()) {
      prc.addModelCount();
    }

    return prc;
  }

  /**
   * Creates a {@link PredictionRearrangerCreator} for creation of a {@link ColumnRearranger} that
   * can be used to predict with a regression random forest.
   *
   * @param dataSpec the spec of the table to predict
   * @param modelSpec the spec of the (regression) random forest
   * @param model the (regression) random forest
   * @param modelRowSamples row samples used to train the individual trees (may be null)
   * @param targetColumnData the target column (may be null)
   * @param config for the prediction
   * @return a creator that allows to create a rearranger for prediction with a random forest
   * @throws InvalidSettingsException if <b>dataSpec</b> is missing some columns the model needs
   */
  public static PredictionRearrangerCreator createPRCForRegressionRF(
      final DataTableSpec dataSpec,
      final TreeEnsembleModelPortObjectSpec modelSpec,
      final TreeEnsembleModel model,
      final TreeSHAPConfiguration config)
      throws InvalidSettingsException {
    int maxDepth = -1;
    int maxNodes = 0;
    int[] uniqueFeaturesPerTree = null;
    if (model != null && config.isShowExplanation()) {

      final TreeEnsembleStatistics s = new TreeEnsembleStatistics();
      s.initialize(model);
      maxDepth = s.getMaximumDepth();
      if (config.isComputeInteractions()) {
        maxNodes = s.getMaximumNumberOfNodes();
        uniqueFeaturesPerTree =
            s.getUniqueFeaturesPerTree(
                model.getNrModels(),
                Math.min(maxNodes, modelSpec.getTableSpec().getNumColumns() - 1));
      }
    }
    final ExplanationRegressionPredictor predictor =
        new ExplanationRegressionPredictor(
            model, modelSpec, dataSpec, maxDepth, maxNodes, uniqueFeaturesPerTree);
    final PredictionRearrangerCreator prc = new PredictionRearrangerCreator(dataSpec, predictor);
    prc.addRegressionPrediction(config.getPredictionColumnName());
    prc.addPredictionVariance(config.getPredictionColumnName());
    if (config.isAppendModelCount()) {
      prc.addModelCount();
    }
    if (config.isShowExplanation()) {
      if (config.isComputeInteractions()) {
        prc.addPredictionItemParser(new InteractionItemParser(modelSpec.getTableSpec()));
      } else {
        prc.addPredictionItemParser(new ExplanationItemParser(modelSpec.getTableSpec()));
      }
    }
    return prc;
  }

  /**
   * Creates a row converter for random forest and gradient boosted trees models.
   *
   * @param modelSpec the spec of the model
   * @param model the actual model (may be null)
   * @param tableSpec the table on which to predict
   * @return a row converter
   * @throws InvalidSettingsException if columns required by <b>modelSpec</b> are not present in
   *     <b>tableSpec</b>
   */
  public static Function<DataRow, PredictorRecord> createRowConverter(
      final TreeEnsembleModelPortObjectSpec modelSpec,
      final AbstractTreeEnsembleModel model,
      final DataTableSpec tableSpec)
      throws InvalidSettingsException {
    final int[] filterIndices = modelSpec.calculateFilterIndices(tableSpec);
    final DataTableSpec learnSpec = modelSpec.getLearnTableSpec();
    return new Function<DataRow, PredictorRecord>() {
      @Override
      public PredictorRecord apply(final DataRow r) {
        return model.createPredictorRecord(new FilterColumnRow(r, filterIndices), learnSpec);
      }
    };
  }

  public static Map<String, Integer> createTargetValueToIndexMap(
      final Map<String, DataCell> targetValueMap) {
    final Map<String, Integer> targetValueToIndexMap = new HashMap<>(targetValueMap.size());
    final Iterator<String> targetValIterator = targetValueMap.keySet().iterator();
    for (int i = 0; i < targetValueMap.size(); i++) {
      targetValueToIndexMap.put(targetValIterator.next(), i);
    }
    return targetValueToIndexMap;
  }

  /**
   * Computes expectations and max depth. Returns max depth and expectations in the map.
   *
   * @param node
   * @param expectations
   * @return
   */
  private static int descendRegressionTree(
      final Map<TreeNodeSignature, Double> coefficientMap,
      final TreeNodeRegression node,
      final int depth,
      final Map<AbstractTreeNode, Double> expectations) {
    int maxDepth = 0;
    if (node.getNrChildren() > 0) {
      assert node.getNrChildren() == 2
          : "ERROR: Tree not binary. Please ensure that you only activate binary splits.";
      final TreeNodeRegression left = node.getChild(0);
      final TreeNodeRegression right = node.getChild(1);
      final int depthLeft = descendRegressionTree(coefficientMap, left, depth + 1, expectations);
      final int depthRight = descendRegressionTree(coefficientMap, right, depth + 1, expectations);
      final double leftWeight = node.getChild(0).getTotalSum();
      final double rightWeight = node.getChild(1).getTotalSum();
      if (leftWeight + rightWeight == 0) {
        expectations.put(node, 0D);
      } else {
        final double leftValue = expectations.get(left);
        final double rightValue = expectations.get(right);
        final double newValue =
            (leftWeight * leftValue + rightWeight * rightValue) / (leftWeight + rightWeight);
        expectations.put(node, newValue);
      }
      maxDepth = Math.max(depthLeft, depthRight) + 1;
    } else {
      expectations.put(node, coefficientMap.get(node.getSignature()));
    }
    return maxDepth;
  }

  /**
   * Descends the tree starting from a node and returns the length of the longest path to a child.
   *
   * @param node
   * @return
   */
  private static int descendTree(final AbstractTreeNode node) {
    if (node.getNrChildren() <= 0) {
      return 0;
    }
    return node.getChildren().stream().map(TreeSHAPUtil::descendTree).max(Integer::compare).get()
        + 1;
  }


  public static int getMaxDepth(final TreeEnsembleModel forest) {
    return IntStream.range(0, forest.getNrModels()) // .parallel()
        .map(
            new IntUnaryOperator() {
              @Override
              public int applyAsInt(final int n) {
                return descendTree(forest.getTreeModel(n).getRootNode());
              }
            })
        .max()
        .getAsInt();
  }

  /**
   * Setups the PredictionRearrangerCreator for classification gbts.
   *
   * @throws InvalidSettingsException if something goes wrong
   */
  public static PredictionRearrangerCreator setupRearrangerCreatorGBT(
      final DataTableSpec dataSpec,
      final TreeEnsembleModelPortObjectSpec modelSpec,
      final MultiClassGradientBoostedTreesModel model,
      final TreeSHAPConfiguration config)
      throws InvalidSettingsException {
    final Map<String, DataCell> targetValueMap = modelSpec.getTargetColumnPossibleValueMap();
    int positiveClassIndex = 0;
    // This is also called during spec generation where the target value may not be
    // known.
    if (targetValueMap != null) {
      final Map<String, Integer> targetVal2Idx = createTargetValueToIndexMap(targetValueMap);
      positiveClassIndex = targetVal2Idx.get(config.getPositiveClass().toString());
    }
    int maxDepth = -1;
    int maxNodes = 0;
    int[] uniqueFeaturesPerTree = null;
    List<Map<AbstractTreeNode, Double>> treeMaps = null;
    if (model != null && config.isShowExplanation()) {
      final TreeEnsembleStatistics s = new TreeEnsembleStatistics();
      s.initialize(model);
      // TODO: Optimize the tree traversals.
      final Pair<Integer, List<Map<AbstractTreeNode, Double>>> pair =
          computeExpectationsClassificationGBT(model, positiveClassIndex);
      maxDepth = pair.first;
      treeMaps = pair.second;
      if (config.isComputeInteractions()) {
        maxNodes = s.getMaximumNumberOfNodes();
        uniqueFeaturesPerTree =
            s.getUniqueFeaturesPerTree(
                model.getNrModels(),
                Math.min(maxNodes, modelSpec.getTableSpec().getNumColumns() - 1));
      }
    }
    // Probabilities are only calculated if they need to be shown.
    final ExplanationGBTPredictor predictor =
        new ExplanationGBTPredictor(
            model,
            createRowConverter(modelSpec, model, modelSpec.getTableSpec()),
            modelSpec.getTableSpec(),
            config.isAppendClassConfidences() || config.isAppendPredictionConfidence(),
            positiveClassIndex,
            maxDepth,
            treeMaps,
            maxNodes,
            uniqueFeaturesPerTree);
    final PredictionRearrangerCreator crc = new PredictionRearrangerCreator(dataSpec, predictor);
    if (config.isAppendClassConfidences()) {
      addClassProbabilites(crc, "P (", modelSpec, model, config);
    }
    crc.addClassPrediction(config.getPredictionColumnName());
    if (config.isAppendPredictionConfidence()) {
      crc.addPredictionConfidence(config.getPredictionColumnName() + CONFIDENCE_SUFFIX);
    }
    if (config.isShowExplanation()) {
      if (config.isComputeInteractions()) {
        crc.addPredictionItemParser(new InteractionItemParser(modelSpec.getTableSpec()));
      } else {
        crc.addPredictionItemParser(new ExplanationItemParser(modelSpec.getTableSpec()));
      }
    }
    return crc;
  }

  /**
   * Setups the PredictionRearrangerCreator for regression gbts.
   *
   * @throws InvalidSettingsException if something goes wrong
   */
  public static PredictionRearrangerCreator setupRearrangerCreatorRegressionGBT(
      final DataTableSpec dataSpec,
      final TreeEnsembleModelPortObjectSpec modelSpec,
      final GradientBoostedTreesModel model,
      final TreeSHAPConfiguration config)
      throws InvalidSettingsException {
    int maxDepth = -1;
    int maxNodes = 0;
    int[] uniqueFeaturesPerTree = null;
    List<Map<AbstractTreeNode, Double>> treeMaps = null;
    if (model != null && config.isShowExplanation()) {
      final TreeEnsembleStatistics s = new TreeEnsembleStatistics();
      s.initialize(model);
      // TODO: Optimize the tree traversals.
      final Pair<Integer, List<Map<AbstractTreeNode, Double>>> pair =
          computeExpectationsRegressionGBT(model);
      maxDepth = pair.first;
      treeMaps = pair.second;
      if (config.isComputeInteractions()) {
        maxNodes = s.getMaximumNumberOfNodes();
        uniqueFeaturesPerTree =
            s.getUniqueFeaturesPerTree(
                model.getNrModels(),
                Math.min(maxNodes, modelSpec.getTableSpec().getNumColumns() - 1));
      }
    }
    // Probabilities are only calculated if they need to be shown.
    final ExplanationGBTRegressionPredictor predictor =
        new ExplanationGBTRegressionPredictor(
            model,
            createRowConverter(modelSpec, model, dataSpec),
            dataSpec,
            treeMaps,
            maxDepth,
            maxNodes,
            uniqueFeaturesPerTree);
    final PredictionRearrangerCreator crc = new PredictionRearrangerCreator(dataSpec, predictor);
    crc.addRegressionPrediction(config.getPredictionColumnName());
    if (config.isShowExplanation()) {
      if (config.isComputeInteractions()) {
        crc.addPredictionItemParser(new InteractionItemParser(modelSpec.getTableSpec()));
      } else {
        crc.addPredictionItemParser(new ExplanationItemParser(modelSpec.getTableSpec()));
      }
    }
    return crc;
  }


  private TreeSHAPUtil() {
    // utility class
  }
}
