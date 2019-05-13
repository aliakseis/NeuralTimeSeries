// https://msdn.microsoft.com/magazine/mt826350

#include <iostream>

#include <vector>

#include <random>

using std::vector;
using std::cout;

namespace NeuralTimeSeries
{
  //class NeuralTimeSeriesProgram
  //{

    static vector<vector<double>> Normalize(const vector<vector<double>>& data)
    {
      // divide all by 100.0
      int rows = data.size();
      int cols = data[0].size();
      //double[][] result = new double[rows][];
      vector<vector<double>> result(rows);
      for (int i = 0; i < rows; ++i)
          result[i].resize(cols);
        //result[i] = new double[cols];

      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j] = data[i][j] / 100.0;
      return result;
    }

    //static double[][] MakeSineData()
    //{
    //  double[][] sineData = new double[17][];
    //  sineData[0] = new double[] { 0.0000, 0.841470985, 0.909297427, 0.141120008 };
    //  sineData[1] = new double[] { 0.841470985, 0.909297427, 0.141120008, -0.756802495 };
    //  sineData[2] = new double[] { 0.909297427, 0.141120008, -0.756802495, -0.958924275 };
    //  sineData[3] = new double[] { 0.141120008, -0.756802495, -0.958924275, -0.279415498 };
    //  sineData[4] = new double[] { -0.756802495, -0.958924275, -0.279415498, 0.656986599 };
    //  sineData[5] = new double[] { -0.958924275, -0.279415498, 0.656986599, 0.989358247 };
    //  sineData[6] = new double[] { -0.279415498, 0.656986599, 0.989358247, 0.412118485 };
    //  sineData[7] = new double[] { 0.656986599, 0.989358247, 0.412118485, -0.544021111 };
    //  sineData[8] = new double[] { 0.989358247, 0.412118485, -0.544021111, -0.999990207 };
    //  sineData[9] = new double[] { 0.412118485, -0.544021111, -0.999990207, -0.536572918 };
    //  sineData[10] = new double[] { -0.544021111, -0.999990207, -0.536572918, 0.420167037 };
    //  sineData[11] = new double[] { -0.999990207, -0.536572918, 0.420167037, 0.990607356 };
    //  sineData[12] = new double[] { -0.536572918, 0.420167037, 0.990607356, 0.65028784 };
    //  sineData[13] = new double[] { 0.420167037, 0.990607356, 0.65028784, -0.287903317 };
    //  sineData[14] = new double[] { 0.990607356, 0.65028784, -0.287903317, -0.961397492 };
    //  sineData[15] = new double[] { 0.65028784, -0.287903317, -0.961397492, -0.750987247 };
    //  sineData[16] = new double[] { -0.287903317, -0.961397492, -0.750987247, 0.14987721 };
    //  return sineData;
    //} // MakeSineData

    static vector<vector<double>> GetAirlineData()
    {
        return {
      { 112., 118., 132., 129., 121 },
      { 118., 132., 129., 121., 135 },
      { 132., 129., 121., 135., 148 },
      { 129., 121., 135., 148., 148 },
      { 121., 135., 148., 148., 136 },
      { 135., 148., 148., 136., 119 },
      { 148., 148., 136., 119., 104 },
      { 148., 136., 119., 104., 118 },
      { 136., 119., 104., 118., 115 },
      { 119., 104., 118., 115., 126 },
       { 104., 118., 115., 126., 141 },
       { 118., 115., 126., 141., 135 },
       { 115., 126., 141., 135., 125 },
       { 126., 141., 135., 125., 149 },
       { 141., 135., 125., 149., 170 },
       { 135., 125., 149., 170., 170 },
       { 125., 149., 170., 170., 158 },
       { 149., 170., 170., 158., 133 },
       { 170., 170., 158., 133., 114 },
       { 170., 158., 133., 114., 140 },
       { 158., 133., 114., 140., 145 },
       { 133., 114., 140., 145., 150 },
       { 114., 140., 145., 150., 178 },
       { 140., 145., 150., 178., 163 },
       { 145., 150., 178., 163., 172 },
       { 150., 178., 163., 172., 178 },
       { 178., 163., 172., 178., 199 },
       { 163., 172., 178., 199., 199 },
       { 172., 178., 199., 199., 184 },
       { 178., 199., 199., 184., 162 },
       { 199., 199., 184., 162., 146 },
       { 199., 184., 162., 146., 166 },
       { 184., 162., 146., 166., 171 },
       { 162., 146., 166., 171., 180 },
       { 146., 166., 171., 180., 193 },
       { 166., 171., 180., 193., 181 },
       { 171., 180., 193., 181., 183 },
       { 180., 193., 181., 183., 218 },
       { 193., 181., 183., 218., 230 },
       { 181., 183., 218., 230., 242 },
       { 183., 218., 230., 242., 209 },
       { 218., 230., 242., 209., 191 },
       { 230., 242., 209., 191., 172 },
       { 242., 209., 191., 172., 194 },
       { 209., 191., 172., 194., 196 },
       { 191., 172., 194., 196., 196 },
       { 172., 194., 196., 196., 236 },
       { 194., 196., 196., 236., 235 },
       { 196., 196., 236., 235., 229 },
       { 196., 236., 235., 229., 243 },
       { 236., 235., 229., 243., 264 },
       { 235., 229., 243., 264., 272 },
       { 229., 243., 264., 272., 237 },
       { 243., 264., 272., 237., 211 },
       { 264., 272., 237., 211., 180 },
       { 272., 237., 211., 180., 201 },
       { 237., 211., 180., 201., 204 },
       { 211., 180., 201., 204., 188 },
       { 180., 201., 204., 188., 235 },
       { 201., 204., 188., 235., 227 },
       { 204., 188., 235., 227., 234 },
       { 188., 235., 227., 234., 264 },
       { 235., 227., 234., 264., 302 },
       { 227., 234., 264., 302., 293 },
       { 234., 264., 302., 293., 259 },
       { 264., 302., 293., 259., 229 },
       { 302., 293., 259., 229., 203 },
       { 293., 259., 229., 203., 229 },
       { 259., 229., 203., 229., 242 },
       { 229., 203., 229., 242., 233 },
       { 203., 229., 242., 233., 267 },
       { 229., 242., 233., 267., 269 },
       { 242., 233., 267., 269., 270 },
       { 233., 267., 269., 270., 315 },
       { 267., 269., 270., 315., 364 },
       { 269., 270., 315., 364., 347 },
       { 270., 315., 364., 347., 312 },
       { 315., 364., 347., 312., 274 },
       { 364., 347., 312., 274., 237 },
       { 347., 312., 274., 237., 278 },
       { 312., 274., 237., 278., 284 },
       { 274., 237., 278., 284., 277 },
       { 237., 278., 284., 277., 317 },
       { 278., 284., 277., 317., 313 },
       { 284., 277., 317., 313., 318 },
       { 277., 317., 313., 318., 374 },
       { 317., 313., 318., 374., 413 },
       { 313., 318., 374., 413., 405 },
       { 318., 374., 413., 405., 355 },
       { 374., 413., 405., 355., 306 },
       { 413., 405., 355., 306., 271 },
       { 405., 355., 306., 271., 306 },
       { 355., 306., 271., 306., 315 },
       { 306., 271., 306., 315., 301 },
       { 271., 306., 315., 301., 356 },
       { 306., 315., 301., 356., 348 },
       { 315., 301., 356., 348., 355 },
       { 301., 356., 348., 355., 422 },
       { 356., 348., 355., 422., 465 },
       { 348., 355., 422., 465., 467 },
       { 355., 422., 465., 467., 404 },
       { 422., 465., 467., 404., 347 },
       { 465., 467., 404., 347., 305 },
       { 467., 404., 347., 305., 336 },
       { 404., 347., 305., 336., 340 },
       { 347., 305., 336., 340., 318 },
       { 305., 336., 340., 318., 362 },
       { 336., 340., 318., 362., 348 },
       { 340., 318., 362., 348., 363 },
       { 318., 362., 348., 363., 435 },
       { 362., 348., 363., 435., 491 },
       { 348., 363., 435., 491., 505 },
       { 363., 435., 491., 505., 404 },
       { 435., 491., 505., 404., 359 },
       { 491., 505., 404., 359., 310 },
       { 505., 404., 359., 310., 337 },
       { 404., 359., 310., 337., 360 },
       { 359., 310., 337., 360., 342 },
       { 310., 337., 360., 342., 406 },
       { 337., 360., 342., 406., 396 },
       { 360., 342., 406., 396., 420 },
       { 342., 406., 396., 420., 472 },
       { 406., 396., 420., 472., 548 },
       { 396., 420., 472., 548., 559 },
       { 420., 472., 548., 559., 463 },
       { 472., 548., 559., 463., 407 },
       { 548., 559., 463., 407., 362 },
       { 559., 463., 407., 362., 405 },
       { 463., 407., 362., 405., 417 },
       { 407., 362., 405., 417., 391 },
       { 362., 405., 417., 391., 419 },
       { 405., 417., 391., 419., 461 },
       { 417., 391., 419., 461., 472 },
       { 391., 419., 461., 472., 535 },
       { 419., 461., 472., 535., 622 },
       { 461., 472., 535., 622., 606 },
       { 472., 535., 622., 606., 508 },
       { 535., 622., 606., 508., 461 },
       { 622., 606., 508., 461., 390 },
       { 606., 508., 461., 390., 432 },
        };
    }

    static void ShowMatrix(vector<vector<double>> matrix, int numRows,
      int decimals, bool indices)
    {
      //int len = matrix.size().ToString().size();
      for (int i = 0; i < numRows; ++i)
      {
        if (indices == true)
          cout << "[" << i/*.ToString().PadLeft(len)*/ << "]  ";
        for (int j = 0; j < matrix[i].size(); ++j)
        {
          double v = matrix[i][j];
          if (v >= 0.0)
            cout << " "; // '+'
          //Console.Write(v.ToString("F" + decimals) + "  ");
          cout << v << "  ";
        }
        cout << '\n';
      }

      if (numRows < matrix.size())
      {
        cout << ". . .\n";
        int lastRow = matrix.size() - 1;
        if (indices == true)
          cout << "[" << lastRow/*.ToString().PadLeft(len)*/ << "]  ";
        for (int j = 0; j < matrix[lastRow].size(); ++j)
        {
          double v = matrix[lastRow][j];
          if (v >= 0.0)
            cout << " "; // '+'
          //Console.Write(v.ToString("F" + decimals) + "  ");
          cout << v << "  ";
        }
      }
      cout << "\n\n";
    }

    static void ShowVector(const vector<double>& v, int decimals,
      int lineLen, bool newLine)
    {
      for (int i = 0; i < v.size(); ++i)
      {
        if (i > 0 && i % lineLen == 0) 
            cout << "\n";
        if (v[i] >= 0)
            cout << " ";
        cout << v[i]/*.ToString("F" + decimals) +*/ << " ";
      }
      if (newLine == true)
        cout << '\n';
    }


  //}; // Program

  class NeuralNetwork
  {
  private: 
      int numInput; // number input nodes
    int numHidden;
    int numOutput;

    vector<double> iNodes;
    vector<vector<double>> ihWeights; // input-hidden
    vector<double> hBiases;
    vector<double> hNodes;

    vector<vector<double>> hoWeights; // hidden-output
    vector<double> oBiases;
    vector<double> oNodes;

    //Random rnd;
    std::default_random_engine dre;

  public: 
      NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
      this->numInput = numInput;
      this->numHidden = numHidden;
      this->numOutput = numOutput;

      iNodes.resize(numInput);

      ihWeights = MakeMatrix(numInput, numHidden, 0.0);
      hBiases.resize(numHidden);
      hNodes.resize(numHidden);

      hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
      oBiases.resize(numOutput);
      oNodes.resize(numOutput);

      //this->rnd = new Random(0);
      InitializeWeights(); // all weights and biases
    } // ctor

  private:
      static vector<vector<double>> MakeMatrix(int rows,
      int cols, double v) // helper for ctor, Train
    {
          vector<vector<double>> result(rows);
      for (int r = 0; r < result.size(); ++r)
        result[r].resize(cols);
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j] = v;
      return result;
    }

  private: 
      void InitializeWeights() // helper for ctor
    {
          std::uniform_real_distribution<double> dis(0, 1);
      // initialize weights and biases to small random values
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      vector<double> initialWeights(numWeights);
      for (int i = 0; i < initialWeights.size(); ++i)
        initialWeights[i] = (0.001 - 0.0001) * dis(dre) + 0.0001;
      this->SetWeights(initialWeights);
    }

  public: 
      void SetWeights(const vector<double>& weights)
    {
      // copy serialized weights and biases in weights[] array
      // to i-h weights, i-h biases, h-o weights, h-o biases
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      if (weights.size() != numWeights)
        //throw new Exception("Bad weights array in SetWeights");
          throw std::runtime_error("Bad weights array in SetWeights");

      int k = 0; // points into weights param

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        hBiases[i] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          hoWeights[i][j] = weights[k++];
      for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];
    }

    vector<double> GetWeights()
    {
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      vector<double> result(numWeights);
      int k = 0;
      for (int i = 0; i < ihWeights.size(); ++i)
        for (int j = 0; j < ihWeights[0].size(); ++j)
          result[k++] = ihWeights[i][j];
      for (int i = 0; i < hBiases.size(); ++i)
        result[k++] = hBiases[i];
      for (int i = 0; i < hoWeights.size(); ++i)
        for (int j = 0; j < hoWeights[0].size(); ++j)
          result[k++] = hoWeights[i][j];
      for (int i = 0; i < oBiases.size(); ++i)
        result[k++] = oBiases[i];
      return result;
    }

    vector<double> ComputeOutputs(const vector<double>& xValues)
    {
      vector<double> hSums(numHidden); // hidden nodes sums scratch array
      vector<double> oSums(numOutput); // output nodes sums

      for (int i = 0; i < xValues.size(); ++i) // copy x-values to inputs
        this->iNodes[i] = xValues[i];

      for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
        for (int i = 0; i < numInput; ++i)
          hSums[j] += this->iNodes[i] * this->ihWeights[i][j]; // note +=

      for (int i = 0; i < numHidden; ++i)  // add biases to hidden sums
        hSums[i] += this->hBiases[i];

      for (int i = 0; i < numHidden; ++i)   // apply activation
        this->hNodes[i] = HyperTan(hSums[i]); // hard-coded

      for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
        for (int i = 0; i < numHidden; ++i)
          oSums[j] += hNodes[i] * hoWeights[i][j];

      for (int i = 0; i < numOutput; ++i)  // add biases to output sums
        oSums[i] += oBiases[i];

      //Array.Copy(oSums, this->oNodes, oSums.size());  // really only 1 value
      std::copy(oSums.begin(), oSums.end(), oNodes.begin());

      //double[] retResult = new double[numOutput]; // could define a GetOutputs 
      //Array.Copy(this->oNodes, retResult, retResult.size());
      //return retResult;
      return { this->oNodes.begin(), this->oNodes.begin() + numOutput };
    }

    private:
        static double HyperTan(double x)
    {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return tanh(x);
    }

    static double LogSig(double x)
    {
      if (x < -20.0) return 0.0; // approximation
      else if (x > 20.0) return 1.0;
      else return 1.0 / (1.0 + exp(x));
    }

    public:
        vector<double> Train(const vector<vector<double>>& trainData, int maxEpochs,
      double learnRate)
    {
      // train using back-prop
      // back-prop specific arrays
      vector<vector<double>> hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // hidden-to-output weight gradients
      vector<double> obGrads(numOutput);                   // output bias gradients

      vector<vector<double>> ihGrads = MakeMatrix(numInput, numHidden, 0.0);  // input-to-hidden weight gradients
      vector<double> hbGrads(numHidden);                   // hidden bias gradients

      vector<double> oSignals (numOutput);                  // local gradient output signals
      vector<double> hSignals(numHidden);                  // local gradient hidden node signals

      int epoch = 0;
      vector<double> xValues(numInput); // inputs
      vector<double> tValues(numOutput); // target values
      double derivative = 0.0;
      double errorSignal = 0.0;

      vector<int> sequence(trainData.size());
      for (int i = 0; i < sequence.size(); ++i)
        sequence[i] = i;

      int errInterval = maxEpochs / 5; // interval to check error
      while (epoch < maxEpochs)
      {
        ++epoch;

        if (epoch % errInterval == 0 && epoch < maxEpochs)
        {
          double trainErr = Error(trainData);
          //Console.WriteLine("epoch = " + epoch + "  error = " + trainErr.ToString("F4"));
          cout << "epoch = " << epoch << "  error = " << trainErr << '\n';
        }

        Shuffle(sequence); // visit each training data in random order
        for (int ii = 0; ii < trainData.size(); ++ii)
        {
          int idx = sequence[ii];
          //Array.Copy(trainData[idx], xValues, numInput);
          std::copy(trainData[idx].begin(), trainData[idx].begin() + numInput, xValues.begin());
          //Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
          std::copy(trainData[idx].begin() + numInput, trainData[idx].begin() + numInput + numOutput, tValues.begin());
          ComputeOutputs(xValues); // copy xValues in, compute outputs 

          // indices: i = inputs, j = hiddens, k = outputs

          // 1. compute output node signals (assumes softmax)
          for (int k = 0; k < numOutput; ++k)
          {
            errorSignal = tValues[k] - oNodes[k];  // Wikipedia uses (o-t)
            derivative = 1.0;  // for Identity activation
            oSignals[k] = errorSignal * derivative;
          }

          // 2. compute hidden-to-output weight gradients using output signals
          for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
              hoGrads[j][k] = oSignals[k] * hNodes[j];

          // 2b. compute output bias gradients using output signals
          for (int k = 0; k < numOutput; ++k)
            obGrads[k] = oSignals[k] * 1.0; // dummy assoc. input value

          // 3. compute hidden node signals
          for (int j = 0; j < numHidden; ++j)
          {
            derivative = (1 + hNodes[j]) * (1 - hNodes[j]); // for tanh
            double sum = 0.0; // need sums of output signals times hidden-to-output weights
            for (int k = 0; k < numOutput; ++k)
            {
              sum += oSignals[k] * hoWeights[j][k]; // represents error signal
            }
            hSignals[j] = derivative * sum;
          }

          // 4. compute input-hidden weight gradients
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihGrads[i][j] = hSignals[j] * iNodes[i];

          // 4b. compute hidden node bias gradients
          for (int j = 0; j < numHidden; ++j)
            hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

          // == update weights and biases

          // update input-to-hidden weights
          for (int i = 0; i < numInput; ++i)
          {
            for (int j = 0; j < numHidden; ++j)
            {
              double delta = ihGrads[i][j] * learnRate;
              ihWeights[i][j] += delta; // would be -= if (o-t)
            }
          }

          // update hidden biases
          for (int j = 0; j < numHidden; ++j)
          {
            double delta = hbGrads[j] * learnRate;
            hBiases[j] += delta;
          }

          // update hidden-to-output weights
          for (int j = 0; j < numHidden; ++j)
          {
            for (int k = 0; k < numOutput; ++k)
            {
              double delta = hoGrads[j][k] * learnRate;
              hoWeights[j][k] += delta;
            }
          }

          // update output node biases
          for (int k = 0; k < numOutput; ++k)
          {
            double delta = obGrads[k] * learnRate;
            oBiases[k] += delta;
          }

        } // each training item

      } // while
      //double[] bestWts = GetWeights();
      return GetWeights();
    } // Train

    private:
        void Shuffle(vector<int>& sequence) // instance method
    {
      for (int i = 0; i < sequence.size(); ++i)
      {
        std::uniform_int_distribution<int> di(i, sequence.size() - 1);
        //int r = this->rnd.Next(i, sequence.size());
        const int r = di(dre);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    } // Shuffle

    private:
        double Error(const vector<vector<double>>& trainData)
    {
      // average squared error per training item
      double sumSquaredError = 0.0;
      vector<double> xValues(numInput); // first numInput values in trainData
      vector<double> tValues(numOutput); // last numOutput values

      // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
      for (int i = 0; i < trainData.size(); ++i)
      {
        //Array.Copy(trainData[i], xValues, numInput);
        std::copy(trainData[i].begin(), trainData[i].begin() + numInput, xValues.begin());
        //Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
        std::copy(trainData[i].begin() + numInput, trainData[i].begin() + numInput + numOutput, tValues.begin());
        vector<double> yValues = ComputeOutputs(xValues); // outputs using current weights
        for (int j = 0; j < numOutput; ++j)
        {
          double err = tValues[j] - yValues[j];
          sumSquaredError += err * err;
        }
      }
      return sumSquaredError / trainData.size();
    } // MeanSquaredError

    public:
        double Accuracy(const vector<vector<double>>& testData, double howClose)
    {
      // percentage correct using winner-takes all
      int numCorrect = 0;
      int numWrong = 0;
      vector<double> xValues(numInput); // inputs
      vector<double> tValues(numOutput); // targets

      for (int i = 0; i < testData.size(); ++i)
      {
        //Array.Copy(testData[i], xValues, numInput); // get x-values
        std::copy(testData[i].begin(), testData[i].begin() + numInput, xValues.begin());
        //Array.Copy(testData[i], numInput, tValues, 0, numOutput); // get t-values
        std::copy(testData[i].begin() + numInput, testData[i].begin() + numInput + numOutput, tValues.begin());
        auto yValues = ComputeOutputs(xValues);

        if (fabs(yValues[0] - tValues[0]) < howClose)  // within 30
          ++numCorrect;
        else
          ++numWrong;

      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

  }; // class NeuralNetwork

} // ns

int main()
{
    using namespace NeuralTimeSeries;

    try
    {

    cout << "\nBegin neural network times series demo\n"
        << "Goal is to predict airline passengers over time\n"
        << "Data from January 1949 to December 1960 \n\n";

    auto trainData = GetAirlineData();
    trainData = Normalize(trainData);
    cout << "Normalized training data:\n";
    ShowMatrix(trainData, 5, 2, true);  // first 5 rows, 2 decimals, show indices

    int numInput = 4; // number predictors
    int numHidden = 12;
    int numOutput = 1; // regression

    cout << "Creating a " << numInput << "-" << numHidden <<
        "-" << numOutput << " neural network";
    NeuralNetwork nn(numInput, numHidden, numOutput);

    int maxEpochs = 10000;
    double learnRate = 0.01;
    cout << "\nSetting maxEpochs = " << maxEpochs << '\n';
    cout << "Setting learnRate = " << learnRate << '\n';

    cout << "\nStarting training\n";
    auto weights = nn.Train(trainData, maxEpochs, learnRate);
    cout << "Done\n";
    cout << "\nFinal neural network model weights and biases:\n\n";
    ShowVector(weights, 2, 10, true);

    double trainAcc = nn.Accuracy(trainData, 0.30);  // within 30
    cout << "\nModel accuracy (+/- 30) on training data = " << trainAcc << '\n';

    //double[] predictors = new double[] { 5.08, 4.61, 3.90, 4.32 };
    auto forecast = nn.ComputeOutputs({ 5.08, 4.61, 3.90, 4.32 });  // 4.33362252510741
    cout << "\nPredicted passengers for January 1961 (t=145): \n";
    cout << (forecast[0] * 100) << '\n';

    //double[] predictors = new double[] { 4.61, 3.90, 4.32, 4.33362252510741 };
    //double[] forecast = nn.ComputeOutputs(predictors);  // 4.33933519590564
    //Console.WriteLine(forecast[0]);

    //double[] predictors = new double[] { 3.90, 4.32, 4.33362252510741, 4.33933519590564 };
    //double[] forecast = nn.ComputeOutputs(predictors);  // 4.69036205766231
    //Console.WriteLine(forecast[0]);

    //double[] predictors = new double[] { 4.32, 4.33362252510741, 4.33933519590564, 4.69036205766231 };
    //double[] forecast = nn.ComputeOutputs(predictors);  // 4.83360378041341
    //Console.WriteLine(forecast[0]);

    //double[] predictors = new double[] { 4.33362252510741, 4.33933519590564, 4.69036205766231, 4.83360378041341 };
    //double[] forecast = nn.ComputeOutputs(predictors);  // 5.50703476366623
    //Console.WriteLine(forecast[0]);

    //double[] predictors = new double[] { 4.33933519590564, 4.69036205766231, 4.83360378041341, 5.50703476366623 };
    //double[] forecast = nn.ComputeOutputs(predictors);  // 6.39605763609294
    //Console.WriteLine(forecast[0]);

    //double[] predictors = new double[] { 4.69036205766231, 4.83360378041341, 5.50703476366623, 6.39605763609294 };
    //double[] forecast = nn.ComputeOutputs(predictors);  // 6.06664881070054
    //Console.WriteLine(forecast[0]);

    //double[] predictors = new double[] { 4.83360378041341, 5.50703476366623, 6.39605763609294, 6.06664881070054 };
    //double[] forecast = nn.ComputeOutputs(predictors);  // 4.95781531728514
    //Console.WriteLine(forecast[0]);

    //double[] predictors = new double[] { 5.50703476366623, 6.39605763609294, 6.06664881070054, 4.95781531728514 };
    //double[] forecast = nn.ComputeOutputs(predictors);  // 4.45837470369601
    //Console.WriteLine(forecast[0]);


    cout << "\nEnd time series demo\n";
    //Console.ReadLine();
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal: " << ex.what() << '\n';
        //Console.ReadLine();
    }
} // Main
