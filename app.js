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

let nn = new NeuralNetwork(2, 80, 1);
const setup = () => {
  


 for (let i = 0; i<1000; i++)
 {
  let data = (training_data[Math.floor(Math.random()*4)]);

  nn.train(data.inputs,data.targets);
 }
 console.log(Math.round(nn.feedForward([1,0])));

}




const btn = document.createElement('button');
btn.innerText = 'Click Me';
btn.addEventListener('click', setup);
document.body.appendChild(btn);




