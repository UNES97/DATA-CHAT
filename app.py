import os
import tempfile
import json
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from pandasai import PandasAI
from pandasai.llm import OpenAI
import sqlite3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class DataChatApp:
    def __init__(self):
        self.df = None
        self.data_source = None
        self.llm = OpenAI(api_token=OPENAI_API_KEY)
        self.pandas_ai = PandasAI(self.llm, verbose=True)
        self.chat_history = []
        self.temp_files = []
        self.db_connection = None
    
    def load_file(self, file):
        """Load data from uploaded file"""
        if file is None:
            return "No file uploaded", None, None
        
        file_path = file.name
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.df = pd.read_csv(file_path)
            elif file_ext == '.xlsx' or file_ext == '.xls':
                self.df = pd.read_excel(file_path)
            elif file_ext == '.json':
                self.df = pd.read_json(file_path)
            else:
                return f"Unsupported file format: {file_ext}", None, None
            
            self.data_source = f"File: {file_name}"
            preview = self.df.head().to_html()
            info = self._get_dataframe_info()
            return f"Loaded successfully: {file_name}", preview, info
        except Exception as e:
            return f"Error loading file: {str(e)}", None, None
    
    def connect_database(self, connection_string, query):
        """Connect to database using connection string"""
        try:
            if connection_string.startswith('sqlite:'):
                # For SQLite, create a temporary database if it's a memory connection
                if 'memory' in connection_string:
                    self.db_connection = sqlite3.connect(':memory:')
                else:
                    db_path = connection_string.replace('sqlite:///', '')
                    self.db_connection = sqlite3.connect(db_path)
            else:
                # For PostgreSQL, MySQL, etc.
                self.db_connection = create_engine(connection_string)
            
            if not query:
                return "Please provide a SQL query", None, None
            
            self.df = pd.read_sql(query, self.db_connection)
            self.data_source = f"Database: {connection_string.split('://')[0]}"
            preview = self.df.head().to_html()
            info = self._get_dataframe_info()
            return "Database connected successfully", preview, info
        except Exception as e:
            return f"Database connection error: {str(e)}", None, None
    
    def _get_dataframe_info(self):
        """Get information about the dataframe"""
        if self.df is None:
            return None
        
        info = {
            "Shape": self.df.shape,
            "Columns": list(self.df.columns),
            "Data Types": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "Missing Values": self.df.isnull().sum().to_dict()
        }
        return json.dumps(info, indent=2)
    
    def chat_with_data(self, query, history):
        """Process natural language query against the loaded data"""
        if self.df is None:
            return "Please load data first before querying.", history
        
        if not query:
            return "Please enter a query.", history
        
        try:
            # Add the user query to history
            history.append((query, ""))
            
            # Process the query using PandasAI
            response = self.pandas_ai(self.df, prompt=query)
            
            # Check if response contains a figure from matplotlib
            if isinstance(response, plt.Figure):
                # Save figure to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                response.savefig(temp_file.name)
                temp_file.close()
                self.temp_files.append(temp_file.name)
                
                # Update the response with the path to the figure
                response = f"<img src='file={temp_file.name}' alt='Visualization' />"
            
            # If response is a dataframe, convert it to an HTML table
            elif isinstance(response, pd.DataFrame):
                response = f"<div style='overflow-x: auto;'>{response.to_html(index=False)}</div>"
            
            # Update the last response in history
            history[-1] = (query, str(response))
            
            return response, history
        except Exception as e:
            history[-1] = (query, f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}", history
    
    def create_visualization(self, viz_type, x_axis, y_axis, title):
        """Create visualization based on user selection"""
        if self.df is None:
            return "Please load data first before creating visualizations."
        
        if not x_axis or (viz_type != 'pie' and not y_axis):
            return "Please select both X and Y axis for the visualization."
        
        try:
            # Check if selected columns exist
            if x_axis not in self.df.columns:
                return f"Column '{x_axis}' not found in the data."
            
            if viz_type != 'pie' and y_axis not in self.df.columns:
                return f"Column '{y_axis}' not found in the data."
            
            fig = None
            
            # Create visualization based on the type
            if viz_type == 'bar':
                fig = px.bar(self.df, x=x_axis, y=y_axis, title=title or f"Bar Chart: {y_axis} by {x_axis}")
            
            elif viz_type == 'line':
                fig = px.line(self.df, x=x_axis, y=y_axis, title=title or f"Line Chart: {y_axis} over {x_axis}")
            
            elif viz_type == 'scatter':
                fig = px.scatter(self.df, x=x_axis, y=y_axis, title=title or f"Scatter Plot: {y_axis} vs {x_axis}")
            
            elif viz_type == 'pie':
                # For pie charts, we use the x_axis as names and can either count occurrences or use y_axis as values
                if y_axis:
                    fig = px.pie(self.df, names=x_axis, values=y_axis, title=title or f"Pie Chart: {y_axis} by {x_axis}")
                else:
                    # Count occurrences of each category in x_axis
                    counts = self.df[x_axis].value_counts().reset_index()
                    counts.columns = ['category', 'count']
                    fig = px.pie(counts, names='category', values='count', title=title or f"Pie Chart: Distribution of {x_axis}")
            
            elif viz_type == 'histogram':
                fig = px.histogram(self.df, x=x_axis, title=title or f"Histogram: Distribution of {x_axis}")
            
            # Return as HTML
            if fig:
                return fig.to_html(include_plotlyjs='cdn', full_html=False)
            else:
                return "Failed to create visualization."
                
        except Exception as e:
            return f"Error creating visualization: {str(e)}"
    
    def generate_summary_cards(self):
        """Generate summary cards (KPIs) for numerical columns"""
        if self.df is None:
            return "Please load data first before generating summary cards."
        
        try:
            # Get numerical columns
            num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not num_cols:
                return "No numerical columns found for summary cards."
            
            cards_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>"
            
            for col in num_cols:
                mean_val = self.df[col].mean()
                median_val = self.df[col].median()
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                
                card_html = f"""
                <div style='background-color: #f5f5f5; border-radius: 5px; padding: 15px; min-width: 200px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='margin-top: 0; color: #333;'>{col}</h3>
                    <p><strong>Mean:</strong> {mean_val:.2f}</p>
                    <p><strong>Median:</strong> {median_val:.2f}</p>
                    <p><strong>Min:</strong> {min_val:.2f}</p>
                    <p><strong>Max:</strong> {max_val:.2f}</p>
                </div>
                """
                cards_html += card_html
            
            cards_html += "</div>"
            return cards_html
        
        except Exception as e:
            return f"Error generating summary cards: {str(e)}"
    
    def cleanup(self):
        """Clean up temporary files"""
        for file in self.temp_files:
            try:
                if os.path.exists(file):
                    os.unlink(file)
            except Exception:
                pass
        
        # Close database connection if exists
        if self.db_connection is not None:
            try:
                if hasattr(self.db_connection, 'close'):
                    self.db_connection.close()
                elif hasattr(self.db_connection, 'dispose'):
                    self.db_connection.dispose()
            except Exception:
                pass

def create_interface():
    app = DataChatApp()
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Data Chat App") as interface:
        gr.Markdown("""
        # 📊 Data Chat Application
        Upload your data file or connect to a database, then chat with your data using natural language!
        """)
        
        with gr.Tabs():
            with gr.TabItem("Load Data"):
                with gr.Tab("File Upload"):
                    file_input = gr.File(label="Upload CSV, Excel, or JSON file")
                    file_upload_button = gr.Button("Load File")
                    file_result = gr.Textbox(label="Result")
                
                with gr.Tab("Database Connection"):
                    conn_str = gr.Textbox(
                        label="Connection String", 
                        placeholder="E.g., sqlite:///data.db, postgresql://user:pass@localhost/db"
                    )
                    query = gr.Textbox(
                        label="SQL Query", 
                        placeholder="SELECT * FROM your_table LIMIT 1000"
                    )
                    db_connect_button = gr.Button("Connect to Database")
                    db_result = gr.Textbox(label="Result")
                
                preview = gr.HTML(label="Data Preview")
                info = gr.JSON(label="Data Information")
            
            with gr.TabItem("Chat with Data"):
                chat_interface = gr.Chatbot(height=400)
                query_input = gr.Textbox(
                    label="Ask a question about your data",
                    placeholder="E.g., Show me the trend of sales over time",
                    lines=2
                )
                chat_button = gr.Button("Ask")
            
            with gr.TabItem("Visualize Data"):
                with gr.Row():
                    with gr.Column(scale=1):
                        viz_type = gr.Dropdown(
                            choices=["bar", "line", "scatter", "pie", "histogram"],
                            label="Visualization Type"
                        )
                        x_axis = gr.Dropdown(label="X-Axis / Category")
                        y_axis = gr.Dropdown(label="Y-Axis / Values (Optional for Pie & Histogram)")
                        viz_title = gr.Textbox(label="Chart Title (Optional)")
                        viz_button = gr.Button("Generate Visualization")
                    
                    with gr.Column(scale=2):
                        viz_output = gr.HTML(label="Visualization")
            
            with gr.TabItem("Summary Stats"):
                summary_button = gr.Button("Generate Summary Cards")
                summary_output = gr.HTML(label="Summary Statistics")
        
        # Set up event handlers
        file_upload_button.click(
            app.load_file, 
            inputs=[file_input], 
            outputs=[file_result, preview, info]
        )
        
        db_connect_button.click(
            app.connect_database, 
            inputs=[conn_str, query], 
            outputs=[db_result, preview, info]
        )
        
        chat_button.click(
            app.chat_with_data, 
            inputs=[query_input, chat_interface], 
            outputs=[query_input, chat_interface]
        )
        
        # Update dropdown options when data is loaded
        def update_column_options():
            if app.df is not None:
                return gr.Dropdown.update(choices=list(app.df.columns))
            return gr.Dropdown.update(choices=[])
        
        file_upload_button.click(update_column_options, outputs=[x_axis])
        file_upload_button.click(update_column_options, outputs=[y_axis])
        db_connect_button.click(update_column_options, outputs=[x_axis])
        db_connect_button.click(update_column_options, outputs=[y_axis])
        
        viz_button.click(
            app.create_visualization, 
            inputs=[viz_type, x_axis, y_axis, viz_title], 
            outputs=[viz_output]
        )
        
        summary_button.click(
            app.generate_summary_cards, 
            outputs=[summary_output]
        )
        
        # Cleanup on close
        interface.on_close(app.cleanup)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)