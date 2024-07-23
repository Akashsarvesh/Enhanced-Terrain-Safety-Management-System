### Multithreaded Video Analysis Application with Streamlit and Twilio Integration

This application allows for multithreaded video analysis using YOLO models to detect accidents, wild animals, and floods. It also sends alerts via Twilio when specific detections occur. The application has two main pages: Analysis and Video Player.

#### Requirements

1. **Python Packages**:
    - streamlit
    - opencv-python
    - opencv-python-headless
    - numpy
    - twilio
    - ultralytics
    - Pillow

2. **Other Tools**:
    - YOLO models for accident detection (`accident7epochs.pt`), wild animal detection (`animal.pt`), and flood detection (`Flood.pt`).

3. **Twilio Account**:
    - Twilio Account SID
    - Twilio Auth Token
    - Twilio phone number
    - Recipient phone number

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-repo/multithreaded-video-analysis.git
    cd multithreaded-video-analysis
    ```

2. **Install the required packages**:
    ```sh
    pip install streamlit opencv-python opencv-python-headless numpy twilio ultralytics Pillow
    ```

3. **Download the YOLO models** and place them in the appropriate directory:
    - `accident.pt` for accident detection
    - `animal.pt` for animal detection
    - `Flood.pt` for flood detection

### Usage

1. **Set up Twilio Credentials**: Update the `account_sid`, `auth_token`, `twilio_number`, and `recipient_number` variables in the `main()` function with your Twilio credentials.

2. **Run the Application**:
    ```sh
    streamlit run app.py
    ```

### Application Structure

- **`main()` Function**: Manages the main page selection and controls the flow between the Analysis and Video Player pages.
- **File Upload and Save**: `save_uploaded_file()` function handles file uploads and saves the video file to a temporary directory.
- **Twilio Alerts**: `send_twilio_alert()` function sends alerts using Twilio.
- **Thread Functions**:
    - `thread_1()`: Detects accidents.
    - `thread_2()`: Detects wild animals.
    - `thread_3()`: Detects floods.
- **Run Video**: `run_video()` function plays the uploaded video.

### Analysis Page

1. **Upload a Video File**: Upload a video file in `.mp4` format.
2. **Run Analysis**: When the "Run Analysis" button is clicked:
    - The video file is saved.
    - Three threads are created to analyze the video for accidents, wild animals, and floods.
    - Alerts are sent via Twilio when detections are made.
    - Twilio alert messages are displayed on the page.

### Video Player Page

1. **Upload a Video File**: Upload a video file in `.mp4` format.
2. **Run Video**: When the "Run Video" button is clicked:
    - The video file is saved temporarily.
    - The video is played in the Streamlit app.

### Example Usage

1. **Upload a video file**.
2. **Select the Analysis page and click "Run Analysis"**.
3. **View the Twilio alert messages sent during the analysis**.
4. **Select the Video Player page to play any video file**.

This application provides a comprehensive tool for analyzing videos for specific events and sending alerts in real-time using multithreading and the YOLO object detection models.
