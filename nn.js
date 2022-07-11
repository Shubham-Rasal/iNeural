function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));

}

function d(y) {
    return y * (1 - y);
}

class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) { //2,3,1
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        this.weights_ih1 = new Matrix(hiddenNodes, inputNodes);
        this.weights_h1_h2 = new Matrix(hiddenNodes, hiddenNodes);
        this.weights_h2_o = new Matrix(outputNodes, hiddenNodes);
        this.bias_ih1 = new Matrix(hiddenNodes, 1)
        this.bias_h1_h2 = new Matrix(hiddenNodes, 1)
        this.bias_h2_o = new Matrix(outputNodes, 1);
        this.lr = 0.1;


        this.weights_ih1.randomize();
        this.weights_h1_h2.randomize();
        this.weights_h2_o.randomize();

        this.bias_ih1.randomize();
        this.bias_h1_h2.randomize();
        this.bias_h2_o.randomize();


    }

    feedForward(input_array) {
        let inputs = Matrix.fromArray(input_array);

        let hiddens = Matrix.multiply(this.weights_ih1, inputs);
        hiddens.add(this.bias_ih1);
        hiddens.map(sigmoid);


        let outputs = Matrix.multiply(this.weights_h2_o, hiddens);
        outputs.add(this.bias_h2_o)
        outputs.map(sigmoid);

        return outputs.toArray();
    }

    train(input, targets) {
        let inputs = Matrix.fromArray(input);
        let hiddens = Matrix.multiply(this.weights_ih1, inputs);
        hiddens.add(this.bias_ih1);
        hiddens.map(sigmoid);

        let hiddens_t = Matrix.transpose(hiddens);
        let inputs_t = Matrix.transpose(inputs);

        let outputs = Matrix.multiply(this.weights_h2_o, hiddens);
        outputs.add(this.bias_h2_o);
        outputs.map(sigmoid);


        targets = Matrix.fromArray(targets);

        let output_errors = Matrix.subtract(targets, outputs);

        let weights_h2_o_t = Matrix.transpose(this.weights_h2_o);


        let hidden_errors = Matrix.multiply(weights_h2_o_t, output_errors);


        let gradient = Matrix.map(outputs, d);

        gradient.multiply(output_errors);
        gradient.multiply(this.lr);




        let weights_h2_o_deltas = Matrix.multiply(gradient, hiddens_t);

        this.weights_h2_o.add(weights_h2_o_deltas);

        let hidden_gradient = Matrix.map(hiddens, d);

        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.multiply(this.lr);


        let weights_ih1_deltas = Matrix.multiply(hidden_gradient, inputs_t);


        this.weights_ih1.add(weights_ih1_deltas);
        // adjusting the biases
        this.bias_ih1.add(hidden_gradient);
        this.bias_h2_o.add(gradient);






    }


}