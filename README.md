# Data Chat Application

This is a Gradio-based web application that allows users to chat with their data using natural language. The application supports loading data from various file formats or connecting to databases, and then querying this data through a conversational interface powered by PandasAI.

## Features

- **Data Loading**:
  - Upload CSV, Excel, or JSON files
  - Connect to databases (PostgreSQL, MySQL, SQLite) using connection strings
  
- **Natural Language Querying**:
  - Ask questions about your data in plain English
  - Get responses as text, tables, or visualizations
  
- **Interactive Visualizations**:
  - Bar charts
  - Line graphs
  - Scatter plots
  - Pie charts
  - Histograms
  
- **Summary Statistics**:
  - Generate KPI cards for numerical columns

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/UNES87/DATA-CHAT.git
   cd data-chat-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

5. Run the application:
   ```
   python app.py
   ```

6. Open your browser and navigate to `http://localhost:7860/`.

## Usage Guide

### Loading Data

1. **File Upload**:
   - Navigate to the "Load Data" tab
   - Select "File Upload"
   - Choose your data file (CSV, Excel, or JSON)
   - Click "Load File"

2. **Database Connection**:
   - Navigate to the "Load Data" tab
   - Select "Database Connection"
   - Enter your connection string (e.g., `sqlite:///data.db`, `postgresql://user:pass@localhost/db`)
   - Enter your SQL query (e.g., `SELECT * FROM your_table LIMIT 1000`)
   - Click "Connect to Database"

### Chatting with Your Data

1. Navigate to the "Chat with Data" tab
2. Type your question in natural language (e.g., "What is the average sales by region?", "Show me the trend of revenue over time")
3. Click "Ask" or press Enter
4. View the response in the chat interface

### Creating Visualizations

1. Navigate to the "Visualize Data" tab
2. Select the visualization type
3. Choose the X and Y axis columns
4. (Optional) Enter a title for your chart
5. Click "Generate Visualization"
6. View the generated chart

### Viewing Summary Statistics

1. Navigate to the "Summary Stats" tab
2. Click "Generate Summary Cards"
3. View the KPI cards for numerical columns

## Dependencies

- Gradio
- Pandas
- NumPy
- Matplotlib
- Plotly
- PandasAI
- OpenAI
- SQLAlchemy
- And more (see requirements.txt)

## License

MIT