class Matrix {
    constructor(rows, columns) {
        this.rows = rows;
        this.columns = columns;
        this.data = [];

        for (let i = 0; i < rows; i++) {
            this.data[i] = [];
            for (let j = 0; j < columns; j++) {
                this.data[i][j] = 0;
            }
        }

    }
    static fromArray(input_array) {

        let inputs = new Matrix(input_array.length, 1);
        for (let index = 0; index < input_array.length; index++) {
            inputs.data[index][0] = input_array[index];


        }
        return inputs;
    }

    toArray() {
        let result = []
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.columns; j++) {
                result[i + j] = this.data[i][j];


            }
        }

        return result;
    }


    //Multiply two matrices
    static multiply(a, b) {
        // console.log(a,b)
        if (a.columns !== b.rows)
            return console.error("The rows of a don't match the columns of b");
        let result = new Matrix(a.rows, b.columns);
        // result.rows = a.rows;
        // result.columns = b.columns;

        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < b.columns; j++) {
                let sum = 0;
                for (let k = 0; k < a.columns; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;

    }

    multiply(k) {
        if(k instanceof Matrix)
        {

            for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.columns; j++)
            this.data[i][j] *= k.data[i][j];
        }
        else
        {
            for (let i = 0; i < this.rows; i++)
            for (let j = 0; j < this.columns; j++)
            this.data[i][j] *= k;

        }
    }

    static subtract(a, b) {
        let result = new Matrix(a.rows, a.columns);
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.columns; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];

            }

        }

        return result;

    }

    //Add two matrices
    add(b) {
        if (b instanceof Matrix) {
            for (let i = 0; i < this.rows; i++)

                for (let j = 0; j < this.columns; j++)

                    this.data[i][j] += b.data[i][j];

        }
        else {
            for (let i = 0; i < this.rows; i++)

                for (let j = 0; j < this.columns; j++)

                    this.data[i][j] += b;

        }



    }
    //Gerate a random matrix
    randomize() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.columns; j++) {
                this.data[i][j] = (Math.random() * 2 - 1);
            }
        }

    }

    //Transpose a matrix
    static transpose(a) {
        let result = new Matrix(a.columns, a.rows);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.columns; j++) {
                result.data[j][i] = a.data[i][j];
            }

        }
        return result;
    }

    //Print a matrix
    print() {
        console.table(this.data)
    }


    map(func) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.columns; j++) {
                this.data[i][j] = func(this.data[i][j]);
            }
        }
    }

    static map(a, func) {
        let result = new Matrix(a.rows, a.columns);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.columns; j++) {
                result.data[i][j] = func(a.data[i][j]);
            }
        }
        return result;
    }
}

module.exports = Matrix;
// export default Matrix;