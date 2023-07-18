let training_data =[ 
  {
    inputs:[1,1],
    targets:[1]
  },
  {
    inputs:[0,0],
    targets:[0]
  },
  {
    inputs:[0,1],
    targets:[0]
  },
  {
    inputs:[1,0],
    targets:[0]
  },

]

let nn = new NeuralNetwork(2, 100, 1);
const setup = () => { 

 for (let i = 0; i<10000; i++)
 {
  let data = (training_data[Math.floor(Math.random()*4)]);
  nn.train(data.inputs,data.targets);
 }
 console.table(training_data)
 console.log(Math.round(nn.feedForward([0,0])));

}




const btn = document.createElement('button');
btn.innerText = 'Click Me';
btn.addEventListener('click', setup);
document.body.appendChild(btn);




