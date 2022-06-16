 function sigmoid(x)
 {
    return 1/(1+Math.exp(-x));

 }

class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) { //2,3,1
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        this.weights_ih = new Matrix(hiddenNodes,inputNodes);
        this.weights_ho = new Matrix(outputNodes,hiddenNodes);
        this.bias_ih = new Matrix(hiddenNodes,1)
        this.bias_ho = new Matrix(outputNodes,1); 
        

        this.weights_ih.randomize();
        this.weights_ho.randomize();
        this.bias_ih.randomize();
        this.bias_ho.randomize();

        



    }

    feedForward(input_array)
    {
        let  inputs = Matrix.fromArray(input_array);

        let hiddens = Matrix.multiply(this.weights_ih,inputs);
        hiddens.add(this.bias_ih);
        hiddens.map(sigmoid);


        let outputs = Matrix.multiply(this.weights_ho,hiddens);
        outputs.add(this.bias_ho)
        outputs.map(sigmoid);


     return outputs.toArray();
    }


}