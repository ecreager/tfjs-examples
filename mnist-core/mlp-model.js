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
const TRAIN_STEPS = 100;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;
const optimizer = tf.train.sgd(LEARNING_RATE);

// Variables that we want to optimize
const layer1Weights =
    tf.variable(tf.randomNormal([Math.pow(IMAGE_SIZE, 2), 400], 0, 0.1));
const layer1Bias = tf.variable(tf.zeros([400]));

const layer2Weights =
    tf.variable(tf.randomNormal([400, 200], 0, 0.1));
const layer2Bias = tf.variable(tf.zeros([200]));

const layer3Weights =
    tf.variable(tf.randomNormal([200, LABELS_SIZE], 0, 0.1));
const layer3Bias = tf.variable(tf.zeros([LABELS_SIZE]));

// Loss function
function loss(labels, ys) {
  return tf.losses.softmaxCrossEntropy(labels, ys).mean();
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
