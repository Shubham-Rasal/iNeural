// let training_data =[ 
//   {
//     inputs:[1,0],
//     targets:[1]
//   },
//   {
//     inputs:[0,1],
//     targets:[1]
//   },
//   {
//     inputs:[0,0],
//     targets:[0]
//   },
//   {
//     inputs:[1,1],
//     targets:[0]
//   },

  
  

// ]

const setup = () => {
  let nn = new NeuralNetwork(2, 9, 1);
  

 for (let i = 0; i<10000; i++)
 {
  let data = (training_data[Math.floor(Math.random()*4)]);
  nn.train(data.inputs,data.targets);
 }
 console.log(nn.feedForward([1,0]));
}




setup();



//function to print top 50 criketers in india
function printCricketers()
{
  //api call for criketers
  
}

