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

        let hiddens_1 = Matrix.multiply(this.weights_ih1, inputs);
        hiddens_1.add(this.bias_ih1);
        hiddens_1.map(sigmoid);
        //Second layer
        let hiddens_2 = Matrix.multiply(this.weights_h1_h2, hiddens_1);
        hiddens_2.add(this.bias_h1_h2);
        hiddens_2.map(sigmoid);

        let outputs = Matrix.multiply(this.weights_h2_o, hiddens_2);
        outputs.add(this.bias_h2_o)
        outputs.map(sigmoid);

        return outputs.toArray();
    }

    train(input, targets) {
        let inputs = Matrix.fromArray(input);

        let hiddens_1 = Matrix.multiply(this.weights_ih1, inputs);
        hiddens_1.add(this.bias_ih1);
        hiddens_1.map(sigmoid);

        let hiddens_2 = Matrix.multiply(this.weights_h1_h2, hiddens_1);
        hiddens_2.add(this.bias_h1_h2);
        hiddens_2.map(sigmoid);


        let outputs = Matrix.multiply(this.weights_h2_o, hiddens_1);
        outputs.add(this.bias_h2_o);
        outputs.map(sigmoid);

        //getting transpose
        let hiddens_1_t = Matrix.transpose(hiddens_1);
        let hiddens_2_t = Matrix.transpose(hiddens_2);
        let inputs_t = Matrix.transpose(inputs);

        targets = Matrix.fromArray(targets);

        //calculate error
        let weights_h2_o_t = Matrix.transpose(this.weights_h2_o);
        let output_errors = Matrix.subtract(targets, outputs);
        let hidden_errors_2 = Matrix.multiply(weights_h2_o_t, output_errors);

        let weights_h1_h2_t = Matrix.transpose(this.weights_h1_h2);
        let hidden_errors_1 = Matrix.multiply(weights_h1_h2_t,hidden_errors_2);




        let gradient_h2o = Matrix.map(outputs, d);
        gradient_h2o.multiply(output_errors);
        gradient_h2o.multiply(this.lr);

        let weights_h2_o_deltas = Matrix.multiply(gradient_h2o, hiddens_2_t);
        this.weights_h2_o.add(weights_h2_o_deltas);


        let gradient_h1_h2 = Matrix.map(hiddens_2,d);
        gradient_h1_h2.multiply(this.lr);
        gradient_h1_h2.multiply(hidden_errors_2);

        let weights_h1_h2_deltas = Matrix.multiply(gradient_h1_h2, hiddens_1_t);
        this.weights_h1_h2.add(weights_h1_h2_deltas);


        let gradient_ih1 = Matrix.map(hiddens_1, d);
        gradient_ih1.multiply(hidden_errors_1);
        gradient_ih1.multiply(this.lr);


        let weights_ih1_deltas = Matrix.multiply(gradient_ih1, inputs_t);
        this.weights_ih1.add(weights_ih1_deltas);


        // adjusting the biases
        this.bias_ih1.add(gradient_ih1);
        this.bias_h1_h2.add(gradient_h1_h2);
        this.bias_h2_o.add(gradient_h2o);

    }


}