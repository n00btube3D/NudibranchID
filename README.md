# NudibranchID: Nudibranch Identification using YOLOv11

This is my final year project that focuses on identifying nudibranchs using the YOLOv11 model. It can classify over 1000 species of Nudibranches, with top-1 accuracy of 77.9% and top-5 accuracy of 91%.

Even if the model make mistakes, it can still reliably get the genus correct, narrowing your search on the beautiful nudibranch you observed.

## Setup

To set up the project, follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/n00btube3D/NudibranchID.git
   cd NudibranchID
   ```

2. Create a Python virtual environment:

   ```sh
   python3 -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Data

The images used for training and testing the model are sourced from [iNaturalist](https://registry.opendata.aws/inaturalist-open-data).

## Usage

To run the app, use the following command:

```sh
python main.py
```

helper_scripts contains a script for regenerating the Python files for the UI.

## Citations

[1] iNaturalist Licensed Observation Images was accessed on DATE from https://registry.opendata.aws/inaturalist-open-data.

[2] Jocher, G., Qiu, J., & Chaurasia, A. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics
