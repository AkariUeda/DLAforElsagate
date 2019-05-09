/* Analyzes the data classification executed by a 2-class
 * LibSVM-based classifier.
 *
 * @author Daniel Moreira (daniel.moreira@ic.unicamp.br) */

#include <math.h>
#include <algorithm>
#include <string>
#include <set>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

/** Data structure that holds the obtained results. */
struct result {
	int actualLabel, predictedLabel;
	double positiveScore, negativeScore;
};

/** Sorts a list of results by the means of their positive scores,
 *  in descending order.  */
bool compareResult(result r1, result r2) {
	return (r1.positiveScore > r2.positiveScore);
}

/** Assesses the desired values, from the obtained results:
 *  - Number of true positive cases;
 *  - Number of true negative cases;
 *  - Number of false positive cases;
 *  - Number of false negative cases;
 *  - Total number of cases;
 *  - Absolute classification accuracy;
 *  - Mean classification accuracy;
 *  - AUC value;
 *  - Top <topN> MAP value (<topN> = 0, for ALL).
 *
 *  The path of the file output by the LibSVM classification
 *  shall be informed.
 *
 *  Parameter <firstNegativeElementLineNumber> brings the number of
 *  the first line in the output file that contains the first of all the
 *  elements belonging to the negative class.
 *
 *  So, it is expected that until such line, all the previous elements belong
 *  to the positive class, while from such line, all the subsequent elements
 *  exclusively belong to the negative class; otherwise assessed rates
 *  will be severely incorrect. */
void assessResults(string *libSVMOutputFilePath,
		int firstNegativeElementLineNumber, int topN) {
	// output file reader
	ifstream libSVMOutputFile;
	libSVMOutputFile.open(libSVMOutputFilePath->data());

	// holds the lines read from the file
	string line;
	getline(libSVMOutputFile, line); // consumes 1st line (useless message)

	// list of results
	vector<result> results;

	// while there are lines to be read
	while (getline(libSVMOutputFile, line)) {
		// mounts the current result
		stringstream lineStream;
		lineStream << line;

		result currentResult;
		lineStream >> currentResult.predictedLabel
				>> currentResult.positiveScore >> currentResult.negativeScore;

		// position-based true label definition
		if (results.size() + 1 < firstNegativeElementLineNumber)
			currentResult.actualLabel = 1;
		else
			currentResult.actualLabel = 0;

		// adds the result to the list of results
		results.push_back(currentResult);
	}

	// closes the output file
	libSVMOutputFile.close();

	// sorts the list of results regarding their positive scores
	// (in descending order)
	sort(results.begin(), results.end(), compareResult);

	// assessing...
	// true positive, true negative, false positive and false negative values
	int truePositiveCount = 0, trueNegativeCount = 0, falsePositiveCount = 0,
			falseNegativeCount = 0;

	// counts the number of actually positive and negative cases
	int positiveCount = 0, negativeCount = 0;

	// absolute and mean classification accuracies
	double absAccuracy, meanAccuracy;

	// MAP calculus
	int top = 0, i = 1, rec = 0;
	double map = 0.0;

	// for each obtained result
	for (int k = 0; k < results.size(); k++) {
		// current result
		result currentResult = results.at(k);

		// tp, tn, fp, fn
		if (currentResult.actualLabel == 1) {
			positiveCount++;

			if (currentResult.predictedLabel == 1)
				truePositiveCount++;
			else
				falseNegativeCount++;
		} else {
			negativeCount++;

			if (currentResult.predictedLabel == -1)
				trueNegativeCount++;
			else
				falsePositiveCount++;
		}

		// topN MAP
		if (k < topN || topN == 0) {
			if (currentResult.actualLabel == 1) {
				top++;
				rec++;
				map = map + top * pow(i, -1);
			}
			i++;
		}
	}

	// calculation of accuracies
	absAccuracy = (truePositiveCount + trueNegativeCount)
			* pow(results.size(), -1);
	meanAccuracy = (truePositiveCount * pow(positiveCount, -1)
			+ trueNegativeCount * pow(negativeCount, -1)) / 2.0;

	// AUC calculation
	double auc;
	{
		// obtained and sorted thresholds
		set<double> thresholds;
		for (int i = 0; i < results.size(); i++)
			thresholds.insert(results.at(i).positiveScore);

		// hit rates and false alarm rates
		vector<double> hitRates(thresholds.size(), 0.0);
		vector<double> falseAlarmRates(thresholds.size(), 0.0);

		int i = 0;
		for (set<double>::iterator iter = thresholds.begin();
				iter != thresholds.end(); ++iter) {
			// current threshold
			double currentThreshold = *iter;

			// current hit rate and false alarm
			for (int j = 0; j < results.size(); j++)
				// if the score if of interest
				if (results.at(j).positiveScore >= currentThreshold) {
					// hit rate update
					if (results.at(j).actualLabel == 1)
						hitRates.at(i) = hitRates.at(i) + 1;

					// false alarm rate update
					else if (results.at(j).actualLabel == 0)
						falseAlarmRates.at(i) = falseAlarmRates.at(i) + 1;
				}

			if (positiveCount > 0)
				hitRates.at(i) = hitRates.at(i) * pow(positiveCount, -1);
			if (negativeCount > 0)
				falseAlarmRates.at(i) = falseAlarmRates.at(i)
						* pow(negativeCount, -1);

			i++;
		}

		// AUC
		auc = 0;
		for (int i = 1; i < falseAlarmRates.size(); i++)
			auc = auc
					+ fabs(falseAlarmRates.at(i) - falseAlarmRates.at(i - 1))
							* hitRates.at(i);
	}

	// values output
	cout << fixed;
	cout << "Results----------" << endl;
	cout << "   True Positive: " << truePositiveCount << endl;
	cout << "   True Negative: " << trueNegativeCount << endl;
	cout << "  False Positive: " << falsePositiveCount << endl;
	cout << "  False Negative: " << falseNegativeCount << endl << endl;

	cout << "           Total: " << results.size() << endl;
	cout << "    Abs Accuracy: " << absAccuracy << endl;
	cout << "   Mean Accuracy: " << meanAccuracy << endl << endl;

	cout << "             AUC: " << auc << endl << endl;

	cout << " MAP (top " << topN << "): " << (rec != 0 ? map * pow(rec, -1) : 0)
			<< endl;
}

/** Turns it into an executable file. */
int main(int paramCount, char** params) {
	cout << "*** libsvm_prediction_analyser Execution. *** " << endl;

	// main parameters
	string libSVMOutputFilePath = ""; 		// -i parameter
	int firstNegativeElementLineNumber = 0; // -n parameter
	int topN = 0;							// -m parameter

	try {
		if (paramCount <= 1)
			throw -1;

		// gathering of parameters
		for (int i = 1; i < paramCount; i = i + 2) {
			stringstream currentParameterStream;
			currentParameterStream << params[i] << params[i + 1];

			char parameterType;
			currentParameterStream >> parameterType >> parameterType;

			switch (parameterType) {
			case 'i':
				currentParameterStream >> libSVMOutputFilePath;
				if (libSVMOutputFilePath.length() <= 0) {
					cerr << "Please verify the -i parameter." << endl;
					throw -2;
				}
				break;

			case 'n':
				currentParameterStream >> firstNegativeElementLineNumber;
				if (firstNegativeElementLineNumber < 1) {
					cerr
							<< "The -n parameter must be equal or greater than ONE."
							<< endl;
					throw -3;
				}
				break;

			case 'm':
				topN = -1; // invalid value
				currentParameterStream >> topN;
				if (topN < 0) {
					cerr
							<< "The -m parameter must be equal or greater than ZERO."
							<< endl;
					throw -4;
				}
				break;

			default:
				throw -5;
			}
		}

		// treatment of mandatory parameters
		if (libSVMOutputFilePath.length() <= 0) {
			cerr << "Please verify the -i parameter." << endl;
			throw -2;
		} else if (firstNegativeElementLineNumber < 1) {
			cerr << "The -n parameter must be equal or greater than ONE."
					<< endl;
			throw -3;
		}

		// logging the parameters, if they are ok
		cout << "Parameters:" << endl << " -i: " << libSVMOutputFilePath << endl
				<< " -n: " << firstNegativeElementLineNumber << endl << " -m: "
				<< topN << endl;

		// parameters are ok
		assessResults(&libSVMOutputFilePath, firstNegativeElementLineNumber,
				topN);
	} catch (int e) {
		cerr
				<< "Usage (with option parameters in any order): libsvm_prediction_analyser"
				<< endl << " -i libsvm_predict_file_path" << endl
				<< " -n first_negative_element_line_number (get 1)" << endl
				<< " -m map_top_n (get 0, all: 0, default: 0)" << endl;
		return e;
	}

	// everything went ok
	cout << "*** Acabou! *** " << endl;
	return 0;
}

