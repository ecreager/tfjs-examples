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

const statusElement = document.getElementById('status');
const messageElement = document.getElementById('message');
const imagesElement = document.getElementById('images');

export function isTraining(mode) {
  statusElement.innerText = 'Training ' + mode + '...';
}
export function trainingLog(message) {
  messageElement.innerText = `${message}\n`;
  console.log(message);
}

export function showTestResults(batch, listOfPreds, labels, mode) {
  statusElement.innerText = 'Testing ' + mode + '...';

  const numPreds = listOfPreds.length;
  const testExamples = batch.xs.shape[0];
  let totalCorrect = 0;
  for (let i = 0; i < testExamples; i++) {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

    const div = document.createElement('div');
    div.className = 'pred-container';

    const canvas = document.createElement('canvas');
    draw(image.flatten(), canvas);

    const pred = document.createElement('div');

    const preds = [];
    for (let j = 0; j < numPreds; j++) {
      preds.push(listOfPreds[j][i]);
    }
    const prediction = preds[0];
    //const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;
    // TODO: compute fraction of correct predictions per input-output pair
    if (correct) {
      totalCorrect++;
    }
    
    // TODO: use frac of correct preds to label this as 'pred-correct', 'pred-incorrect', or 'pred-partially-correct'
    pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
    pred.innerText = `p: ${preds.toString()}`;

    div.appendChild(pred);
    div.appendChild(canvas);

    imagesElement.appendChild(div);
  }

  const accuracy = 100 * totalCorrect / testExamples;
  const displayStr =
      `accuracy: ${accuracy.toFixed(2)}% (${totalCorrect} / ${testExamples})`;
  messageElement.innerText = `${displayStr}\n`;
  console.log(displayStr);
}

export function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}
