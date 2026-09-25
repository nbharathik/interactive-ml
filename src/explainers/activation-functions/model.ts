/** Everything the run is built from, derived from the params once; the page and the lesson tests share it. */

import { getClassificationDataset, splitData } from '../../lib/datasets/points';
import { getRegressionDataset } from '../../lib/datasets/regression';
import type { PointData, RegressionData } from '../../lib/datasets/types';
import { activation } from '../../lib/ml/activations';
import type { ActivationFn, ActivationName } from '../../lib/ml/activations';
import { loss as lossOf } from '../../lib/ml/losses';
import type { LossFn, LossName } from '../../lib/ml/losses';
import { dataFromPoints, dataFromRegression, valueScale } from '../../lib/ml/mlpTrainer';
import type { TrainerConfig, TrainerData, ValueScale } from '../../lib/ml/mlpTrainer';
import type { ActParams } from './config';

/** Training loss under which a run counts as fitted. */
export const FITTED_CLASS = 0.03;
export const FITTED_REGRESS = 0.02;

/** Params that rebuild the model; the rest only change the picture. */
export const MODEL_KEYS: readonly (keyof ActParams & string)[] = [
  'task',
  'pointsDataset',
  'curveDataset',
  'count',
  'noise',
  'seed',
  'activation',
  'leak',
  'classLoss',
  'regLoss',
  'huberDelta',
  'depth',
  'width',
  'learningRate',
  'initScale',
  'batchSize',
];

export interface ActModel {
  classify: boolean;
  pointData: PointData;
  curveData: RegressionData;
  classCount: number;
  lossName: LossName;
  lossFn: LossFn;
  act: ActivationFn;
  scale: ValueScale;
  data: TrainerData;
  config: TrainerConfig;
  fitted: number;
}

export function buildModel(params: ActParams): ActModel {
  const classify = params.task !== 'regress';
  const pointData = getClassificationDataset(params.pointsDataset).generate({ count: params.count, noise: params.noise, seed: params.seed });
  const curveData = getRegressionDataset(params.curveDataset).generate({ count: params.count, noise: params.noise, seed: params.seed });
  const classCount = classify ? pointData.classCount : 2;
  // A binary loss has one output, so more than two classes always take softmax.
  const lossName = (classify ? (classCount > 2 ? 'softmaxCE' : params.classLoss) : params.regLoss) as LossName;
  const lossFn = lossOf(lossName, { huberDelta: params.huberDelta, classCount });
  const act = activation(params.activation as ActivationName, { leakSlope: params.leak });
  const scale = valueScale(curveData.points);
  let data: TrainerData;
  if (classify) {
    const split = splitData(pointData.points, 0.75, params.seed);
    data = dataFromPoints(split.train, split.test, lossFn, classCount);
  } else {
    const split = splitData(curveData.points, 0.75, params.seed);
    data = dataFromRegression(split.train, split.test, curveData.xRange, scale);
  }
  const config: TrainerConfig = {
    task: classify ? 'classify' : 'regress',
    depth: params.depth,
    width: params.width,
    activation: params.activation as ActivationName,
    activationParams: { leakSlope: params.leak },
    loss: lossName,
    lossParams: { huberDelta: params.huberDelta },
    classCount,
    init: params.activation === 'relu' || params.activation === 'leakyRelu' || params.activation === 'elu' ? 'he' : 'xavier',
    initScale: params.initScale,
    seed: params.seed,
    optimiser: { name: 'adam' },
    learningRate: params.learningRate,
    batchSize: params.batchSize,
  };
  return { classify, pointData, curveData, classCount, lossName, lossFn, act, scale, data, config, fitted: classify ? FITTED_CLASS : FITTED_REGRESS };
}
