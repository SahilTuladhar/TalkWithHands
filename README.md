# TalkwithHands: A Sign Language Detection System

![TalkwithHands Logo](https://github.com/SahilTuladhar/TalkWithHands/blob/new-approach/images/Talkwithhands%20logo.png)

TalkwithHands is a robust sign language detection system designed to interpret gestures accurately in real-time. By leveraging the power of MediaPipe for landmark extraction and deep learning models for gesture recognition, this system provides an interactive solution for sign language interpretation.

## Key Features 👌🤝

- Custom Dataset Creation: Manually captured frames from video sequences to ensure high-quality data for training.
- Landmark Feature Extraction: Utilized MediaPipe to extract key pose, face, and hand landmarks.
- Deep Learning Models: Implemented both LSTM and CNN-LSTM (LRCN) architectures to capture spatial and temporal gesture patterns.
- Real-time Integration: Integrated the trained models into a real-time system for gesture recognition.

#### 1. Data Collection

- Dataset Creation: Captured 30 frames per video for 100 videos per action using MediaPipe Holistic.
- Landmark Feature Extraction:
  - Extracted 468 face, 21 left-hand, and 21 right-hand landmarks.
  - Normalized and reshaped data into unified arrays for efficient processing.

#### 2. Data Preprocessing

- Converted frames to RGB and normalized landmarks.
- Stored processed data in MongoDB for efficient managements.

#### 3. Model Development

- LSTM Model:
  - Captures temporal gesture dynamics.
  - Input prepared as a sequence of reshaped landmark arrays.
- CNN-LSTM (LRCN) Model:
  - Combines CNNs for spatial feature extraction and LSTMs for sequential learning.
  - Ensures stability and robustness in gesture recognition.

#### 4. Model Training and Evaluation

- Split dataset into training, validation, and testing sets.
- Techniques applied:
  - Batch normalization and dropout regularization.
  - Adaptive learning rate optimization.
- Evaluation Metrics:
  - Confusion matrix.
  - Accuracy comparison across epochs.
  - Loss minimization using visualization tools like Matplotlib.

#### 5. Real-time Integration

- Captures live video feed using OpenCV.
- Processes gestures in sequences of 30 frames.
- Implements a cooldown mechanism for consistent predictions.

![TalkwithHands Workflow](https://github.com/SahilTuladhar/TalkWithHands/blob/new-approach/images/Talkwithhands%20development%20chart.jpg)

![Data Create](https://github.com/SahilTuladhar/TalkWithHands/blob/new-approach/images/Dataset%20Creation%20Flowchart.jpg)

![Model Training](https://github.com/SahilTuladhar/TalkWithHands/blob/new-approach/images/Talkwithhands%20Model%20Training%20Flowchart.jpg)

## Limitations ⛔️

- Limited gesture vocabulary as dataset created
- Struggles to interpret overlapping or ambiguous gestures.

## Code Requirements 📱

You can install Conda for python which resolves all the dependencies for machine learning.

## Setup 🖥️

### Prerequisites

- Python 3.12 or later.
- Required libraries: TensorFlow, MediaPipe, OpenCV, pandas, MongoDB, matplotlib.

- Create directories to organize the files
- Environmental Setup: Install all the required dependencies such as pandas,numpy,matplotlib etc.
- Development Workflow
  - Files present in branch new_approach
  - Create dataset using Create_dataset.ipynb
  - Preprocess data using Preprocesses_dataset.ipynb
  - Train the LSTM and CNN-LSTM model using seperate LSTM_model_new and CNN_LSTM_model_combined files respectively
  - Real time application of models using Realtime_LRCN.ipynb and Realtime_test_LSTM.ipynb
  - Use Jupyter notebooks for experimentation.

## Execution ▶️

Finally, Run the scripts or files in the correct order using the Jupyter Notebook.

## Results 📈

- Developed a computationally efficient system with custom datasets.
- CNN-LSTM architecture proved highly effective for gesture detection.
- Successfully integrated real-time gesture recognition capabilities.
  ![LRCN](https://github.com/SahilTuladhar/TalkWithHands/blob/new-approach/images/LRCN_result.png)
  ![LSTM](https://github.com/SahilTuladhar/TalkWithHands/blob/new-approach/images/LSTM_result.png)

## Future Work 🔮

- Expand gesture vocabulary to include more actions.
- Improve real-time accuracy for complex gestures and to seperate ambiguous gestures.
- Incorporate advanced models for better generalization.

For detailed information regarding the results please check out the pdf link given
