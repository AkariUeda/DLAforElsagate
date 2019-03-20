/* Extracts and saves frames from a given video file.
 *
 * @author Daniel Moreira (daniel.moreira@ic.unicamp.br) 
 * @author Mauricio Perez (mauricio.perez@students.ic.unicamp.br) */

#include <dirent.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace boost;
using namespace cv;

/** Returns the current date and time. */
string getCurrentDateTime() {
	time_t now = time(0);
	struct tm tstruct;
	char buf[80];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

	return buf;
}

/** Returns the frame rate of a video, given its file path. */
double getVideoFrameRate(string videoFilePath) {
	double frameRate = 0;

	stringstream shellScript;
	shellScript << "ffprobe -i " << videoFilePath
			<< " -v quiet -show_streams -select_streams v 2>&1 | grep 'avg_frame_rate=' | cut -c16-";

	FILE *pipe = popen(shellScript.str().data(), "r");
	if (!pipe) {
		cerr << "ERROR: Could not obtain video frame rate: " << videoFilePath
				<< "." << endl;
		throw -1;
	}

	else {
		char buffer[128];
		string result = "";

		while (!feof(pipe))
			if (fgets(buffer, 128, pipe) != NULL)
				result += buffer;
		pclose(pipe);

		vector<string> frameRateTokens;
		split(frameRateTokens, result, is_any_of("/"));

		double numerator, denominator;
		numerator = atof(frameRateTokens.front().data());
		denominator = atof(frameRateTokens.back().data());
		
		if ( numerator == 0){
			// avg_frame_rate is 0/0, need to compute fps from frame number and duration
			double frame_number, duration;
			
			stringstream shellScript_fn, shellScript_dur;
			
			// Retrieving frame number
			shellScript_fn << "ffprobe -i " << videoFilePath
				<< " -v quiet -show_streams -select_streams v 2>&1 | grep 'nb_frames=' | cut -c11-";
				
			result = "";
			
			pipe = popen(shellScript_fn.str().data(), "r");
			
			if (!pipe) {
				cerr << "ERROR: Could not obtain video frame rate: " << videoFilePath
						<< "." << endl;
				throw -1;
			}
			
			while (!feof(pipe))
				if (fgets(buffer, 128, pipe) != NULL)
					result += buffer;
			pclose(pipe);
			
			split(frameRateTokens, result, is_any_of("/"));
			
			frame_number = atof(frameRateTokens.front().data());
			//~ frame_number = atof(result);
			
			// Retrieving duration
			shellScript_dur << "ffprobe -i " << videoFilePath
				<< " -v quiet -show_streams -select_streams v 2>&1 | grep 'duration=' | cut -c10-";
			
			result = "";
			
			pipe = popen(shellScript_dur.str().data(), "r");
			
			while (!feof(pipe))
				if (fgets(buffer, 128, pipe) != NULL)
					result += buffer;
			pclose(pipe);
			
			split(frameRateTokens, result, is_any_of("/"));
			
			duration = atof(frameRateTokens.front().data());
			//~ duration = atof(result);
			
			// Computing Frame Rate
			frameRate = frame_number / duration;
		}
		else
			frameRate = numerator / denominator;
	}

	return frameRate;
}

/** Reads a list with absolute paths to videos, from a given text file. */
void readVideoFilePathList(string inputFilePath, vector<string> *answer) {
	// input text file reader
	ifstream fileReader;
	fileReader.open(inputFilePath.data());
	if (fileReader.fail()) {
		cerr << "Could not open file " << inputFilePath << "." << endl;
		throw -1;
	}

	// holds the current line read from the input file
	string line;

	// while there are paths to be read, adds them to the answer
	while (getline(fileReader, line)) {
		trim(line);
		if (!line.empty())
			answer->push_back(line);
	}

	// closes the input file
	fileReader.close();
}

/** Determines new width and height values for the frames extracted from a given video file,
 *  of which original dimensions are given as parameters, assuming that the user decided for
 *  a new number of pixels per frame, and that the original video aspect ratio must be
 *  maintained. If the desired new number of pixels per frame is greater than the original
 *  one, this function does nothing (i.e. it simply returns the original video width and
 *  height. */
void calculateNewWidthAndHeight(int originalWidth, int originalHeight,
		int desiredPixelCount, int *newWidth, int *newHeight) {
	int originalPixelCount = originalWidth * originalHeight;

	// calculates the new width and height
	if (desiredPixelCount < originalPixelCount) {
		double videoAspectRatio = originalWidth * pow(originalHeight, -1);

		*newHeight = round(sqrt(desiredPixelCount / videoAspectRatio));
		*newWidth = round(videoAspectRatio * *newHeight);
	}
	// does not change video resolution, if the desired pixel count is greater than
	// the current one
	else {
		*newHeight = originalHeight;
		*newWidth = originalWidth;
	}
}

/** Extracts the solicited frames from a given video, and saves them in the given directory.
 *  It is recommended for the video to be in H.264 MPEG-4 format. The frames are output as
 *  TIF images, named with the video file name + frame number + a sequence number (from
 *  00000001 to N). The size of the saved frames can be informed as a new desired total number
 *  of pixels per frame, or 0 if the original size shall be maintained. If the new desired
 *  total number of pixels is greater than the original one, the sizes of the frames are
 *  simply maintained.
 *
 *  Parameter <extractedFramesPerSecond> defines how many frames are supposed to be
 *  extracted per second of video. */
void extractAndSaveVideoFrames(string videoFilePath,
		double extractedFramesPerSecond, string frameDirPath,
		int totalPixelCount) {
	// obtains the video frame rate, with the help of ffprobe
	double frameRate = getVideoFrameRate(videoFilePath);

	// calculates the frame extraction step
	int frameExtractionStep = int(round(frameRate / extractedFramesPerSecond));

	// obtains the name of the original video file
	vector<string> *videoFilePathTokens = new vector<string>;
	split(*videoFilePathTokens, videoFilePath, is_any_of("/"));
	string videoFileName = videoFilePathTokens->back();
	videoFilePathTokens->clear();
	delete videoFilePathTokens;

	// video reader
	VideoCapture * videoReader = new VideoCapture(videoFilePath);

	// determines new frame width and frame height values, if it is the case
	int frameWidth = 0, frameHeight = 0;
	if (totalPixelCount > 0) {
		Mat firstFrame;
		videoReader->read(firstFrame);

		int originalFrameWidth, originalFrameHeight;
		originalFrameWidth = firstFrame.cols;
		originalFrameHeight = firstFrame.rows;

		calculateNewWidthAndHeight(originalFrameWidth, originalFrameHeight,
				totalPixelCount, &frameWidth, &frameHeight);

		videoReader->release();
		videoReader = new VideoCapture(videoFilePath);
	}

	// obtains the wanted frames, based on the given numbers
	int currentPosition = 0, savedFrameCount = 0, currentWantedFrameNumber = 0, currentWantedFrameNumberNext;
	while (true) {
		// if the current frame is a wanted one
		currentWantedFrameNumberNext = currentWantedFrameNumber + 1;
		if (currentPosition == currentWantedFrameNumber) {
			Mat currentFrame;
			if (!videoReader->read(currentFrame))
				break;

			// if the frame is to be resized, does it
			if (frameWidth > 0 && frameHeight > 0)
				resize(currentFrame, currentFrame,
						Size(frameWidth, frameHeight), CV_INTER_CUBIC);

			// mounts the name of the file of the current frame
			// completes the number of the frame with zeros
			char frameNumberChars[8];
			sprintf(frameNumberChars, "%.7d", currentWantedFrameNumber);

			// completes the position of the frame with zeros
			char framePositionChars[8];
			sprintf(framePositionChars, "%.7d", (savedFrameCount + 1));

			// mounts the name of the current file
			stringstream frameFilePathStream;
			frameFilePathStream << frameDirPath.data() << "/" << videoFileName
					<< "." << frameRate << "fps" << "-" << frameNumberChars
					<< "-" << framePositionChars << ".tif";

			// saves the frame
			imwrite(frameFilePathStream.str(), currentFrame);

			// one more frame saved
			savedFrameCount++;

			// next wanted frame
			//~ currentWantedFrameNumber = currentWantedFrameNumber
					//~ + frameExtractionStep;
		} else if (currentPosition == currentWantedFrameNumberNext) {
			Mat currentFrame;
			if (!videoReader->read(currentFrame))
				break;

			// if the frame is to be resized, does it
			if (frameWidth > 0 && frameHeight > 0)
				resize(currentFrame, currentFrame,
						Size(frameWidth, frameHeight), CV_INTER_CUBIC);

			// mounts the name of the file of the current frame
			// completes the number of the frame with zeros
			char frameNumberChars[8];
			sprintf(frameNumberChars, "%.7d", currentWantedFrameNumberNext);

			// completes the position of the frame with zeros
			char framePositionChars[8];
			sprintf(framePositionChars, "%.7d", (savedFrameCount + 1));

			// mounts the name of the current file
			stringstream frameFilePathStream;
			frameFilePathStream << frameDirPath.data() << "/" << videoFileName
					<< "." << frameRate << "fps" << "-" << frameNumberChars
					<< "-" << framePositionChars << ".tif";

			// saves the frame
			imwrite(frameFilePathStream.str(), currentFrame);

			// one more frame saved
			savedFrameCount++;

			// next wanted frame
			currentWantedFrameNumber = currentWantedFrameNumber
					+ frameExtractionStep;
		}
		// else, just consumes the current frame, to proceed
		else if (!videoReader->grab())
			break;

		// one more position
		currentPosition++;
	}

	// frees memory
	videoReader->release();
	delete videoReader;
}

/** Individually extracts the frames from the videos refereed by the given list of video
 *  file paths. It is recommended for the videos to be in H.264 MPEG-4 format. The frames
 *  are output as TIF images, named with their video file name + frame number + a sequence
 *  number (from 00000001 to N). The size of the saved frames can be informed as a new
 *  desired total number of pixels per frame, or 0 if the original size shall be maintained.
 *  If the new desired total number of pixels is greater than the original one, the sizes of
 *  the frames are simply maintained. The number of threads to let run simultaneously when
 *  extracting the frames must also be informed.
 *
 *  Parameter <extractedFramesPerSecond> defines how many frames are supposed to be
 *  extracted per second of video. */
void extractAndSaveFrames(vector<string> *videoFilePaths, string *frameDirPath,
		int totalPixelCount, double extractedFramesPerSecond,
		int simThreadCount) {
	// time register
	cout << "Begin time: " << getCurrentDateTime() << endl;

	// tries to open the given dir path to store the extracted frames
	DIR *pDir;
	pDir = opendir(frameDirPath->data());
	if (pDir == NULL)
		// tries to create the directory
		mkdir(frameDirPath->data(), 0777);

	pDir = opendir(frameDirPath->data());
	if (pDir == NULL) {
		cerr << "Could not open neither create directory " << frameDirPath
				<< "." << endl;
		throw -1;
	}
	closedir(pDir);

	// holds the number of treated video files
	int filesCount = 0;

	// for each video file path
	for (int i = 0; i < videoFilePaths->size(); i = i + simThreadCount) {
		// current group of up to <simThreadCount> threads
		thread_group descriptionThreadGroup;

		for (int j = 0; j < simThreadCount; j++)
			if (i + j < videoFilePaths->size()) {
				// file path of the current video
				string currentVideoFilePath = videoFilePaths->at(i + j);

				// thread creation to extract  the chosen frames from the current video
				descriptionThreadGroup.add_thread(
						new thread(extractAndSaveVideoFrames,
								currentVideoFilePath, extractedFramesPerSecond,
								*frameDirPath, totalPixelCount));

				// counts one more treated file
				filesCount++;
			} else
				break;

		// executes the current thread group
		descriptionThreadGroup.join_all();

		// logging
		cout << "Progress: treated files " << filesCount << "/"
				<< videoFilePaths->size() << "." << endl;
	}

	// time register
	cout << "End time: " << getCurrentDateTime() << endl;
}

/** Turns it into an executable file. */
int main(int paramCount, char** params) {
	cout << "*** frame_extractor Execution. *** " << endl;

	// main parameters
	string inputFilePath = "";				// -i parameter
	string frameDirPath = "";				// -o parameter
	int totalPixelCount = 0;				// -p parameter
	double extractedFramesPerSecond = 5.0;	// -f parameter
	int simThreadCount = 1;					// -t parameter

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
				currentParameterStream >> inputFilePath;
				if (inputFilePath.length() <= 0) {
					cerr << "Please verify the -i parameter." << endl;
					throw -2;
				}
				break;

			case 'o':
				currentParameterStream >> frameDirPath;
				if (frameDirPath.length() <= 0) {
					cerr << "Please verify the -o parameter." << endl;
					throw -3;
				}
				break;

			case 'p':
				totalPixelCount = -1; // invalid value
				currentParameterStream >> totalPixelCount;
				if (totalPixelCount < 0) {
					cerr
							<< "The -p parameter must be equal or greater than ZERO."
							<< endl;
					throw -4;
				}
				break;

			case 'f':
				extractedFramesPerSecond = 0.0; // invalid value
				currentParameterStream >> extractedFramesPerSecond;
				if (extractedFramesPerSecond <= 0.0) {
					cerr << "The -f parameter must be greater than ZERO."
							<< endl;
					throw -5;
				}
				break;

			case 't':
				simThreadCount = 0; // invalid value
				currentParameterStream >> simThreadCount;
				if (simThreadCount < 1) {
					cerr
							<< "The -t parameter must be equal or greater than ONE."
							<< endl;
					throw -6;
				}
				break;

			default:
				throw -7;
			}
		}

		// treatment of mandatory parameters
		if (inputFilePath.length() <= 0) {
			cerr << "Please verify the -i parameter." << endl;
			throw -2;
		} else if (frameDirPath.length() <= 0) {
			cerr << "Please verify the -o parameter." << endl;
			throw -3;
		}

		// logging the parameters, if they are ok
		cout << "Parameters:" << endl << " -i: " << inputFilePath << endl
				<< " -o: " << frameDirPath << endl << " -p: " << totalPixelCount
				<< endl << " -f: " << extractedFramesPerSecond << endl
				<< " -t: " << simThreadCount << endl;
	} catch (int e) {
		cerr << "Usage (with option parameters in any order): frame_extractor"
				<< endl << " -i input_file_path" << endl << " -o frame_dir_path"
				<< endl << " -p total_pixel_count (get 0, all: 0, default: 0)"
				<< endl
				<< " -f extracted_frames_per_second (gt 0, default: 5.0)"
				<< endl << " -t sim_thread_count (get 1, default: 1)" << endl;
		return e;
	}

	// main execution, if parameters are ok
	// tries to obtain the file names of the videos
	vector<string> videoFilePaths;
	try {
		readVideoFilePathList(inputFilePath, &videoFilePaths);
	} catch (int e) {
		cerr << "Could not obtain the video filenames." << endl;
		return 10 * e;
	}

	// frame extraction
	try {
		extractAndSaveFrames(&videoFilePaths, &frameDirPath, totalPixelCount,
				extractedFramesPerSecond, simThreadCount);
	} catch (int e) {
		cerr << "Could not extract the frames." << endl;
		return 100 * e;
	}

	// everything went ok
	cout << "*** Acabou! ***" << endl;
	return 0;
}
