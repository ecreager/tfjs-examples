/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {MnistData} from './data';

// Hyperparameters.
const LEARNING_RATE = .1;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 1000;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;
const optimizer = tf.train.sgd(LEARNING_RATE);

// Variables that we want to optimize
const eps = tf.scalar(1e-9);
const minLogSigma = -1.0;
const maxLogSigma = -0.5;

const layer1WeightsMu = tf.variable(tf.randomNormal([Math.pow(IMAGE_SIZE, 2), 400], 0, 0.1));
const layer1WeightsLogSigma = tf.variable(tf.randomUniform(layer1WeightsMu.shape, minLogSigma, maxLogSigma));
const layer1WeightsEpsilon = tf.randomNormal(layer1WeightsMu.shape);
const layer1BiasMu = tf.variable(tf.zeros([400]));
const layer1BiasLogSigma = tf.variable(tf.randomUniform(layer1BiasMu.shape, minLogSigma, maxLogSigma));
const layer1BiasEpsilon = tf.randomNormal(layer1BiasMu.shape);

const layer2WeightsMu = tf.variable(tf.randomNormal([400, 200], 0, 0.1));
const layer2WeightsLogSigma = tf.variable(tf.randomUniform(layer2WeightsMu.shape, minLogSigma, maxLogSigma));
const layer2WeightsEpsilon = tf.randomNormal(layer2WeightsMu.shape);
const layer2BiasMu = tf.variable(tf.zeros([200]));
const layer2BiasLogSigma = tf.variable(tf.randomUniform(layer2BiasMu.shape, minLogSigma, maxLogSigma));
const layer2BiasEpsilon = tf.randomNormal(layer2BiasMu.shape);

const layer3WeightsMu = tf.variable(tf.randomNormal([200, LABELS_SIZE], 0, 0.1));
const layer3WeightsLogSigma = tf.variable(tf.randomUniform(layer3WeightsMu.shape, minLogSigma, maxLogSigma));
const layer3WeightsEpsilon = tf.randomNormal(layer3WeightsMu.shape);
const layer3BiasMu = tf.variable(tf.zeros([LABELS_SIZE]));
const layer3BiasLogSigma = tf.variable(tf.randomUniform(layer3BiasMu.shape, minLogSigma, maxLogSigma));
const layer3BiasEpsilon = tf.randomNormal(layer3BiasMu.shape);

// tensors derived (e.g., reparam trick) from variables
function diagonalGaussianEntropy(logSigma) {
  const s = logSigma.shape
  const k = tf.scalar(s[0]*s[1]);
  const sumLogSigmas = tf.sum(logSigma.flatten());
  const doubleEntropy = k.mul(tf.scalar(1.0).add(tf.log(tf.scalar(2.0*Math.PI)))).add(sumLogSigmas);
  return tf.scalar(0.5).mul(doubleEntropy);
}

const layer1WeightsSigma = tf.exp(layer1WeightsLogSigma.add(eps));
const layer1Weights = layer1WeightsMu.add(layer1WeightsSigma.mul(layer1WeightsEpsilon));
const layer1BiasSigma = tf.exp(layer1BiasLogSigma.add(eps));
//const layer1Bias = layer1BiasMu.add(layer1BiasSigma.mul(layer1BiasEpsilon));
const layer1Bias = layer1BiasMu;
//const layer1Entropy = diagonalGaussianEntropy(layer1WeightsLogSigma).add(diagonalGaussianEntropy(layer1WeightsLogSigma));
const layer1Entropy = diagonalGaussianEntropy(layer1WeightsLogSigma)

const layer2WeightsSigma = tf.exp(layer2WeightsLogSigma.add(eps));
const layer2Weights = layer2WeightsMu;
const layer2BiasSigma = tf.exp(layer2BiasLogSigma.add(eps));
//const layer2Bias = layer2BiasMu.add(layer2BiasSigma.mul(layer2BiasEpsilon));
const layer2Bias = layer2BiasMu;
//const layer2Entropy = diagonalGaussianEntropy(layer2WeightsLogSigma).add(diagonalGaussianEntropy(layer2WeightsLogSigma));
const layer2Entropy = diagonalGaussianEntropy(layer2WeightsLogSigma)

const layer3WeightsSigma = tf.exp(layer3WeightsLogSigma.add(eps));
const layer3Weights = layer3WeightsMu.add(layer3WeightsSigma.mul(layer3WeightsEpsilon));
const layer3BiasSigma = tf.exp(layer3BiasLogSigma.add(eps));
//const layer3Bias = layer3BiasMu.add(layer3BiasSigma.mul(layer3BiasEpsilon));
const layer3Bias = layer3BiasMu;
//const layer3Entropy = diagonalGaussianEntropy(layer3WeightsLogSigma).add(diagonalGaussianEntropy(layer3WeightsLogSigma));
const layer3Entropy = diagonalGaussianEntropy(layer3WeightsLogSigma)

const qEntropy = layer1Entropy.add(layer2Entropy).add(layer3Entropy);

// Loss function
function loss(labels, ys) {
  // TODO: evaluate ELBO with > 1 sample
  // TODO: Gaussian normal prior on mus and sigmas
  const logpyIxw = tf.losses.softmaxCrossEntropy(labels, ys).mean().mul(tf.scalar(-1.0));
  //const logqw = qEntropy.mul(tf.scalar(-1.0));
  const elbo = logpyIxw.add(qEntropy);
  return elbo.mul(tf.scalar(-1.0))
}

// Our actual model
function model(inputXs) {
  // layer 1
  const layer1 = tf.tidy(() => {
    return inputXs.matMul(layer1Weights).add(layer1Bias).relu()
  });

  // layer 2
  const layer2= tf.tidy(() => {
    return layer1.matMul(layer2Weights).add(layer2Bias).relu()
  });

  // layer 3
  return layer2.matMul(layer3Weights).add(layer3Bias);
}

// Train the model.
export async function train(data, log) {
  const returnCost = true;

  for (let i = 0; i < TRAIN_STEPS; i++) {
    const cost = optimizer.minimize(() => {
      const batch = data.nextTrainBatch(BATCH_SIZE);
      return loss(batch.labels, model(batch.xs));
    }, returnCost);

    log(`loss[${i}]: ${cost.dataSync()}`);

    await tf.nextFrame();
  }
}

// Predict the digit number from a batch of input images.
export function predict(x) {
  const pred = tf.tidy(() => {
    const axis = 1;
    return model(x).argMax(axis);
  });
  return Array.from(pred.dataSync());
}

// Given a logits or label vector, return the class indices.
export function classesFromLabel(y) {
  const axis = 1;
  const pred = y.argMax(axis);

  return Array.from(pred.dataSync());
}
