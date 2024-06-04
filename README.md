# 🌦️ Weather Data Forecast

This repository contains the development of a [Facebook Prophet](https://github.com/facebook/prophet) model to forecast the average temperature in Valencia (Spain).

## 📁 Project Structure

- `app.py`: Main script.
- `config.py`: File where the API Key is to be stored.
- `utils.py`: Utility functions for data processing and modeling.
- `variables.csv`: File with the description of each variable in the dataset.
- `requirements.txt`: List of dependencies required to run the project.

## 🛠️ Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/jmudy/weather-data-forecast.git
    ```

2. Navigate to the project directory:

    ```bash
    cd weather-data-forecast
    ```

3. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    ```

4. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Get your own API Key on the [AEMET](https://www.aemet.es/) website:

    Enter the following [link](https://opendata.aemet.es/centrodedescargas/altaUsuario) and request an API Key. Once you get it in your email you have to replace it in the `config.py` file.

## 🚀 Usage

1. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501` to interact with the application.
