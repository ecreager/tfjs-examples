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

import {MnistData} from './data';
import * as model from './model';
import * as mlpModel from './mlp-model';
import * as bnnMlpModel from './bnn-mlp-model';
import * as ui from './ui';


// mode: choose from {conv, mlp, bnn-mlp}
const mode = 'bnn-mlp';
function getExportedModel(mode) {
  if (mode == 'conv') {
    return model;
  } else if (mode == 'mlp') {
    return mlpModel; 
  } else if (mode == 'bnn-mlp') {
    return bnnMlpModel; 
  } else {
    throw 'Invalid mode; choose from {conv, mlp, bnn-mlp}'; 
  }
}
const exportedModel = getExportedModel(mode);
export default exportedModel

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

async function train() {
  ui.isTraining(mode);
  await exportedModel.train(data, ui.trainingLog);
}

async function test() {
  const testExamples = 50;
  const numPreds = 4;
  const listOfPreds = [];
  const batch = data.nextTestBatch(testExamples);
  for (var i = 0; i < numPreds; i++) {
    var predictions = exportedModel.predict(batch.xs);
    listOfPreds.push(predictions);
  } 
  //const predictions = exportedModel.predict(batch.xs);
  for (let j = 0; j < numPreds; j++) {
    console.log('pred', j, listOfPreds.toString());
  }

  const labels = exportedModel.classesFromLabel(batch.labels);

  ui.showTestResults(batch, listOfPreds, labels, mode);
  //ui.showTestResults(batch, predictions, labels, mode);
}

async function mnist() {
  await load();
  await train();
  test();
}
mnist();
