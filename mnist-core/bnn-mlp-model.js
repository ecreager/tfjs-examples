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
import {NUM_TRAIN_ELEMENTS} from './data';

// Hyperparameters.
const LEARNING_RATE = .001;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 50;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;
const optimizer = tf.train.sgd(LEARNING_RATE);

// hyperparam values
const minLogSigma = 0.1;
const maxLogSigma = 0.5;
const lambda = tf.scalar(0.0000005);

// Variables that we want to optimize
const layer1WeightsMu = tf.variable(tf.randomNormal([Math.pow(IMAGE_SIZE, 2), 400]));
const layer1WeightsLogSigma = tf.variable(tf.randomUniform(layer1WeightsMu.shape, minLogSigma, maxLogSigma));
//const layer1WeightsEpsilon = tf.randomNormal(layer1WeightsMu.shape);
const layer1BiasMu = tf.variable(tf.zeros([400]));
const layer1BiasLogSigma = tf.variable(tf.randomUniform(layer1BiasMu.shape, minLogSigma, maxLogSigma));
//const layer1BiasEpsilon = tf.randomNormal(layer1BiasMu.shape);

const layer2WeightsMu = tf.variable(tf.randomNormal([400, 200]));
const layer2WeightsLogSigma = tf.variable(tf.randomUniform(layer2WeightsMu.shape, minLogSigma, maxLogSigma));
const layer2WeightsEpsilon = tf.randomNormal(layer2WeightsMu.shape);
const layer2BiasMu = tf.variable(tf.zeros([200]));
const layer2BiasLogSigma = tf.variable(tf.randomUniform(layer2BiasMu.shape, minLogSigma, maxLogSigma));
const layer2BiasEpsilon = tf.randomNormal(layer2BiasMu.shape);

const layer3WeightsMu = tf.variable(tf.randomNormal([200, LABELS_SIZE]));
const layer3WeightsLogSigma = tf.variable(tf.randomUniform(layer3WeightsMu.shape, minLogSigma, maxLogSigma));
const layer3WeightsEpsilon = tf.randomNormal(layer3WeightsMu.shape);
const layer3BiasMu = tf.variable(tf.zeros([LABELS_SIZE]));
const layer3BiasLogSigma = tf.variable(tf.randomUniform(layer3BiasMu.shape, minLogSigma, maxLogSigma));
const layer3BiasEpsilon = tf.randomNormal(layer3BiasMu.shape);

// tensors derived (e.g., reparam trick) from variables
function diagonalGaussianEntropy(logSigma) {
  const n_weights = tf.scalar(IMAGE_SIZE*400 + 400*200 + 200*10);
  const s = logSigma.shape;
  const k = tf.scalar(s[0]*s[1]);
  const sumLogSigmas = tf.sum(logSigma.flatten());
  //console.log('dge', sumLogSigmas.mean().toString());
  const log2pi = tf.log(tf.scalar(2.0).mul(tf.scalar(Math.PI)));
  const constPart = tf.scalar(0.5).mul(k).mul(tf.scalar(1.0).add(log2pi));
  return constPart.add(sumLogSigmas).div(n_weights);
}

function logPrior(w) {
  return tf.scalar(-1.0).mul(tf.sum(tf.pow(w, tf.scalar(2.0))));
}

function reparamTrick(mu, logSigma) {
  return tf.tidy(() => {
    const epsilon = tf.randomNormal(mu.shape);
    //console.log('e', epsilon.toString());
    const sigma = tf.exp(logSigma.add(tf.scalar(1e-9)));
    const w = mu.add(sigma.mul(epsilon));
    //console.log('w', w.toString());
    return w;
  });
}

//const eps = tf.scalar(1e-9);  // TODO: remove
//const layer1WeightsSigma = tf.exp(layer1WeightsLogSigma.add(eps));
//const layer1Weights = layer1WeightsMu.add(layer1WeightsSigma.mul(layer1WeightsEpsilon));
//const layer1BiasSigma = tf.exp(layer1BiasLogSigma.add(eps));
//const layer1Bias = layer1BiasMu.add(layer1BiasSigma.mul(layer1BiasEpsilon));
const layer1Entropy = diagonalGaussianEntropy(layer1WeightsLogSigma).add(diagonalGaussianEntropy(layer1WeightsLogSigma));

//const layer2WeightsSigma = tf.exp(layer2WeightsLogSigma.add(eps));
//const layer2Weights = layer2WeightsMu.add(layer2WeightsSigma.mul(layer2WeightsEpsilon));
//const layer2BiasSigma = tf.exp(layer2BiasLogSigma.add(eps));
//const layer2Bias = layer2BiasMu.add(layer2BiasSigma.mul(layer2BiasEpsilon));
const layer2Entropy = diagonalGaussianEntropy(layer2WeightsLogSigma).add(diagonalGaussianEntropy(layer2WeightsLogSigma));

//const layer3WeightsSigma = tf.exp(layer3WeightsLogSigma.add(eps));
//const layer3Weights = layer3WeightsMu.add(layer3WeightsSigma.mul(layer3WeightsEpsilon));
//const layer3BiasSigma = tf.exp(layer3BiasLogSigma.add(eps));
//const layer3Bias = layer3BiasMu.add(layer3BiasSigma.mul(layer3BiasEpsilon));
const layer3Entropy = diagonalGaussianEntropy(layer3WeightsLogSigma).add(diagonalGaussianEntropy(layer3WeightsLogSigma));

const qEntropy = layer1Entropy.add(layer2Entropy).add(layer3Entropy);

// Loss function
function loss(labels, ys) {
  return tf.tidy(() => {
    // TODO: evaluate ELBO with > 1 sample
    // TODO: Gaussian normal prior on mus and sigmas
    // const NUM_DATASET_ELEMENTS = 65000;
    // const TRAIN_TEST_RATIO = 5 / 6;
    // const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
    // const NOverM = tf.scalar(NUM_TRAIN_ELEMENTS/BATCH_SIZE);
    // const logpyIxw = NOverM.mul(tf.losses.softmaxCrossEntropy(labels, ys).sum().mul(tf.scalar(-1.0)));
    const logpyIxw = tf.losses.softmaxCrossEntropy(labels, ys).mean().mul(tf.scalar(-1.0));
    //const logpyIxw = tf.losses.softmaxCrossEntropy(labels, ys).mean().mul(tf.scalar(-1.0));
    //
    // strictly speaking this is weird b/c we use different epsilons to evaluate the predictions and weight decay penalty
    const layer1Weights = reparamTrick(layer1WeightsMu, layer1WeightsLogSigma);
    const layer2Weights = reparamTrick(layer2WeightsMu, layer2WeightsLogSigma);
    const layer3Weights = reparamTrick(layer3WeightsMu, layer3WeightsLogSigma);
    const logpw = lambda.mul((logPrior(layer1Weights).add(logPrior(layer2Weights)).add(logPrior(layer3Weights))));
    // console.log('l1 avg w', layer1Weights.mean().toString());
    // console.log('l2 avg w', layer2Weights.mean().toString());
    // console.log('l3 avg w', layer3Weights.mean().toString());
    // TODO: print variance of weights
    // console.log('l1 avg logsig', layer1WeightsLogSigma.mean().toString());
    // console.log('l2 avg logsig', layer2WeightsLogSigma.mean().toString());
    // console.log('l3 avg logsig', layer3WeightsLogSigma.mean().toString());
    console.log('pw', logpw.toString());
    console.log('pDIw', logpyIxw.toString());
    console.log('Hq', qEntropy.toString());
    //const logqw = qEntropy.mul(tf.scalar(-1.0));
    const elbo = (logpyIxw.add(logpw).add(qEntropy));
    return elbo.mul(tf.scalar(-1.0))
  });
}

// Our actual model
function model(inputXs) {
  // layer 1
  const layer1 = tf.tidy(() => {
    const layer1Weights = reparamTrick(layer1WeightsMu, layer1WeightsLogSigma);
    const layer1Weights2 = reparamTrick(layer1WeightsMu, layer1WeightsLogSigma);
    const layer1Bias = reparamTrick(layer1BiasMu, layer1BiasLogSigma);
    console.log(1, layer1Weights.norm().toString(), layer1Bias.norm().toString());
    console.log('1mu', layer1WeightsMu.toString());
    console.log('1ls', layer1WeightsLogSigma.toString());
    return inputXs.matMul(layer1Weights).add(layer1Bias).relu()
  });

  // layer 2
  const layer2 = tf.tidy(() => {
    const layer2Weights = reparamTrick(layer2WeightsMu, layer2WeightsLogSigma);
    const layer2Bias = reparamTrick(layer2BiasMu, layer2BiasLogSigma);
    console.log(2, layer2Weights.norm().toString(), layer2Bias.norm().toString());
    return layer1.matMul(layer2Weights).add(layer2Bias).relu()
  });

  // layer 3
  const layer3 = tf.tidy(() => {
    const layer3Weights = reparamTrick(layer3WeightsMu, layer3WeightsLogSigma);
    const layer3Bias = reparamTrick(layer3BiasMu, layer3BiasLogSigma);
    console.log(3, layer3Weights.norm().toString(), layer3Bias.norm().toString());
    return layer2.matMul(layer3Weights).add(layer3Bias).relu()
  });

  return layer3;
}

// Train the model.
export async function train(data, log) {
  const returnCost = true;

  for (let i = 0; i < TRAIN_STEPS; i++) {
    console.log('iter', i);
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
    const yhat = model(x);
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
