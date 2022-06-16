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
        // let outputs = this.feedForward(inputs);
        let inputs = Matrix.fromArray(input);
        // console.table(this.weights_ih.data)
        // console.table(inputs.data)
        let hiddens = Matrix.multiply(this.weights_ih, inputs);
        hiddens.add(this.bias_ih);
        hiddens.map(sigmoid);

        let hiddens_t = Matrix.transpose(hiddens);
        // console.table(this.weights_ho.data)
        // console.table(this.weights_ho.data)
        // console.table(hiddens.data)

        let outputs = Matrix.multiply(this.weights_ho, hiddens);
        outputs.add(this.bias_ho);
        outputs.map(sigmoid);
        // console.table(outputs.data);
        // console.log(outputs.data[0])


        targets = Matrix.fromArray(targets);
        // outputs = Matrix.fromArray(outputs);

        let output_errors = Matrix.subtract(targets, outputs);

        let weights_ho_t = Matrix.transpose(this.weights_ho);
        // console.table(this.weights_ho.data)


        let hidden_errors = Matrix.multiply(weights_ho_t, output_errors);


        let gradient = Matrix.map(outputs, d);
        // let gradient_t = Matrix.transpose(gradient);
        // console.log(gradient_t,output_errors)
        // console.table(gradient.data)
        // console.table(output_errors.data)

        gradient.multiply(output_errors);
        gradient.multiply(this.lr);

        // let weights_ho_deltas = Matrix.multiply(gradient, outputs);
        //  console.table(outputs.data,"Outputs")


         let weights_ho_deltas = Matrix.multiply(gradient, hiddens_t);

         weights_ho_deltas.print();



    }


}