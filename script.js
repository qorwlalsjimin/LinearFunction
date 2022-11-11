console.log('Hello TensorFlow');

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
  const carsData = await carsDataResponse.json();
  const cleaned = carsData.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  }))
  .filter(car => (car.mpg != null && car.horsepower != null));

  return cleaned;
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

  return model;
}



async function run() { //메인함수
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );

  // More code will be added below
  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({name: 'Model Summary'}, model);
}

document.addEventListener('DOMContentLoaded', run);

