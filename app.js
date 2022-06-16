const setup = () => {
  let nn = new NeuralNetwork(2, 3, 3);
  let input = [1, 2];
  let target = [0,2,3];
  let output = nn.feedForward(input);
  // console.table(output);

  nn.train(input, target);
}


setup();