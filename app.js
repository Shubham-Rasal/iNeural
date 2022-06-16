const setup = () => {
   let nn = new NeuralNetwork(2,3,1);
   let input = [1,2];
  let output = nn.feedForward(input); 
  console.log(output);
}


setup();