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

        this.weights_ih = new Matrix(hiddenNodes, inputNodes);
        this.weights_ho = new Matrix(outputNodes, hiddenNodes);
        this.bias_ih = new Matrix(hiddenNodes, 1)
        this.bias_ho = new Matrix(outputNodes, 1);
        this.lr = 0.1;


        this.weights_ih.randomize();
        this.weights_ho.randomize();
        this.bias_ih.randomize();
        this.bias_ho.randomize();





    }

    feedForward(input_array) {
        let inputs = Matrix.fromArray(input_array);

        let hiddens = Matrix.multiply(this.weights_ih, inputs);
        hiddens.add(this.bias_ih);
        hiddens.map(sigmoid);


        let outputs = Matrix.multiply(this.weights_ho, hiddens);
        outputs.add(this.bias_ho)
        outputs.map(sigmoid);


        return outputs.toArray();
    }

    train(input, targets) {
        let inputs = Matrix.fromArray(input);
        let hiddens = Matrix.multiply(this.weights_ih, inputs);
        hiddens.add(this.bias_ih);
        hiddens.map(sigmoid);

        let hiddens_t = Matrix.transpose(hiddens);
        let inputs_t = Matrix.transpose(inputs);

        let outputs = Matrix.multiply(this.weights_ho, hiddens);
        outputs.add(this.bias_ho);
        outputs.map(sigmoid);


        targets = Matrix.fromArray(targets);

        let output_errors = Matrix.subtract(targets, outputs);

        let weights_ho_t = Matrix.transpose(this.weights_ho);


        let hidden_errors = Matrix.multiply(weights_ho_t, output_errors);


        let gradient = Matrix.map(outputs, d);

        gradient.multiply(output_errors);
        gradient.multiply(this.lr);




        let weights_ho_deltas = Matrix.multiply(gradient, hiddens_t);

        this.weights_ho.add(weights_ho_deltas);

        let hidden_gradient = Matrix.map(hiddens, d);

        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.multiply(this.lr);


        let weights_ih_deltas = Matrix.multiply(hidden_gradient, inputs_t);


        this.weights_ih.add(weights_ih_deltas);
        // adjusting the biases
        this.bias_ih.add(hidden_gradient);
        this.bias_ho.add(gradient);






    }


}