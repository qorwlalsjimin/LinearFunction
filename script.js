console.log('Hello TensorFlow');

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
    let rawData = [];
    for(let x = 0; x<=25; x++){
        let obj = {
            "x":x,
            "y":2*x + 1
        }
        rawData.push(obj);
    }
    return rawData;
}

//여기에 코드 붙이기
function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // 이 학습모델에는 인공 신경망이 2개 있음
  // 인공 신경망이 3개 이상이어야 딥러닝으로 볼 수 있음
  // Add a single input layer //인공 신경망1
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

  // Add an output layer //인공 신경망2
  model.add(tf.layers.dense({units: 1, useBias: true}));

  // Add an output layer //인공 신경망2
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) { //tensor은 배열 형식으로 만들어짐
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.x)
    const labels = data.map(d => d.y);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

async function trainModel(model, inputs, labels, epochs) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData, normalizationData, epochs) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {

    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100, 1]));

    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.x, y: d.y,
  }));

  tfvis.render.scatterplot(
    {name: `Check prediction data epochs: ${epochs}`},
    {values: [originalPoints, predictedPoints], series: ['study', 'estimation']},
    {
      xLabel: 'x',
      yLabel: 'y',
      height: 300
    }
  );
}

async function run() { //메인함수
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
    x: d.x,
    y: d.y,
  }));

  tfvis.render.scatterplot(
    {name: 'y = 2x + 1'},
    {values},
    {
      xLabel: 'x',
      yLabel: 'y',
      height: 300
    }
  );

  // More code will be added below
  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({name: 'Model Summary'}, model);

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  // Train the model
  await trainModel(model, inputs, labels, 100);
  testModel(model, data, tensorData, 100);

   // Train the model
   await trainModel(model, inputs, labels, 100);
   testModel(model, data, tensorData, 100+100);

   // Train the model
   await trainModel(model, inputs, labels, 300);
   testModel(model, data, tensorData, 100+200+200);
}

document.addEventListener('DOMContentLoaded', run);

