# Development of Career Counselling Chatbot

## Overview
The Career Counselling Chatbot is an artificial intelligence driven platform designed to deliver personalized and affordable career guidance. By leveraging Natural Language Processing and Machine Learning, the chatbot interacts with users to answer career related queries and offers a tailored Personality Test to predict suitable professions or further study options.

## Features
* **Interactive Chat Interface**: Processes user text inputs using NLP to extract keywords and matches them against a comprehensive dataset of career opportunities.
* **Personalized Personality Test**: Asks a series of situational and interest based questions to gauge user aptitude.
* **Visualized Recommendations**: Generates a dynamic pie chart displaying recommended career paths and professions based on the Personality Test results.
* **Detailed Career Insights**: Provides extensive information and descriptions for the suggested occupations to help users make informed decisions.

## Project Architecture
The system utilizes a structured workflow:
1. **User Input**: The user submits a query through the Streamlit interface.
2. **NLP Processing**: The input is tokenized and processed using NLTK and TF IDF vectorization.
3. **Data Retrieval**: The processed keywords are matched against the dataset.
4. **Response Generation**: The bot returns the most relevant statement or triggers the Personality Test flow.

![System Architecture Workflow](<img width="1004" height="700" alt="image" src="https://github.com/user-attachments/assets/1e9db2de-4334-4e48-b18d-f739e524cd9a" />
)

## Technologies Used
* **Programming Language**: Python
* **Libraries**: NumPy, Pandas, NLTK, Scikit learn, Bokeh, TensorFlow
* **Web Framework**: Streamlit
* **Environment**: Jupyter Notebook and Anaconda

## Installation and Usage
Follow these steps to run the chatbot locally:

1. Install the required dependencies:
`pip install tensorflow bokeh numpy pandas streamlit joblib pathlib nltk`

2. Navigate to the project directory.

3. Run the setup script:
`python setup.py`

4. Launch the Streamlit application:
`streamlit run app.py`

5. To stop the application, press CTRL+C in the terminal.

## Graphical User Interface

### Chatbot Interface
The initial interface greets the user and allows them to ask general career questions.
![Chatbot Greeting Banner](<img width="602" height="561" alt="image" src="https://github.com/user-attachments/assets/fc7374f8-88f0-4503-9aeb-0a6ba9e13168" />
)

### Personality Test
Users can type **start my test** to initiate the assessment. The bot asks for the user education level and presents a series of interactive questions.
![Personality Test Questionnaire](<img width="892" height="443" alt="image" src="https://github.com/user-attachments/assets/56260bb4-7028-4d66-9a25-4d2d0c9507dc" />)
![Personality Test Questionnaire](<img width="1004" height="1048" alt="image" src="https://github.com/user-attachments/assets/70f14de1-7c5c-4e4e-b3b7-239e7fcc41c6" />
)

### Results and Visualization
After completing the test, the application renders a pie chart using the Bokeh library to visualize the top recommended professions along with detailed descriptions for each.
![Results Visualization](<img width="808" height="693" alt="image" src="https://github.com/user-attachments/assets/519bdf64-69f4-4afc-b10d-45bd4ca6d912" />
)
