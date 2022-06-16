const setup = () => {
   let nn = new NeuralNetwork(2,3,1);
   let input = [1,2];
   let target = [1];
  let output = nn.feedForward(input); 
  console.table(output);

  nn.train(input,target);
}


setup();