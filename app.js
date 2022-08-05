let training_data =[ 
  {
    inputs:[1,1],
    targets:[0]
  },
  {
    inputs:[0,1],
    targets:[1]
  },
  {
    inputs:[0,0],
    targets:[0]
  },
  {
    inputs:[1,0],
    targets:[1]
  },

]

const setup = () => {
  let nn = new NeuralNetwork(2, 3, 1);
  

 for (let i = 0; i<10000; i++)
 {
  let data = (training_data[Math.floor(Math.random()*4)]);
  nn.train(data.inputs,data.targets);
 }
 console.table(nn.feedForward([0,1]));
 document.write(nn.feedForward([1,0]))
}


setup();


