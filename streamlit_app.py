import os
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import dask.dataframe as dd
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import gc
from functools import lru_cache
import modin.pandas as mpd

class OptimizedDataLoader:
    """Optimized data loader with caching and parallel processing capabilities"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.parquet_path = self.cache_dir / 'cached_accidents.parquet'
        self.chunk_size = 100_000  # Adjust based on available RAM
        
    def _convert_to_parquet(self):
        """Convert CSV to Parquet format for faster subsequent reads"""
        try:
            # Read CSV in chunks using Dask
            df = dd.read_csv(
                self.file_path,
                assume_missing=True,
                blocksize='64MB',  # Adjust based on available RAM
                dtype={
                    'ID': 'object',
                    'Severity': 'int8',
                    'Start_Lat': 'float32',
                    'Start_Lng': 'float32',
                    'Street': 'object',
                    'City': 'object',
                    'State': 'object',
                    'Weather_Condition': 'object',
                    'Sunrise_Sunset': 'object'
                }
            )
            
            # Convert to Parquet with optimized settings
            df.to_parquet(
                self.parquet_path,
                engine='pyarrow',
                compression='snappy',
                write_index=False
            )
            
            return True
        except Exception as e:
            st.error(f"Error converting to Parquet: {str(e)}")
            return False

    @lru_cache(maxsize=1)
    def _get_date_bounds(self):
        """Get min and max dates from the dataset (cached)"""
        df = dd.read_parquet(self.parquet_path)
        min_date = df['Start_Time'].min().compute()
        max_date = df['Start_Time'].max().compute()
        return min_date, max_date

    def optimize_memory(self, df):
        """Optimize memory usage of the dataframe"""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert string columns to categorical
                df[col] = pd.Categorical(df[col])
            elif df[col].dtype == 'float64':
                # Downcast float64 to float32
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                # Downcast int64 to smallest integer type
                df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df

    def load_filtered_data(self, start_date=None, end_date=None, severity_range=None, time_filter=None):
        """Load and filter data efficiently"""
        try:
            # Convert to Parquet if not already done
            if not self.parquet_path.exists():
                if not self._convert_to_parquet():
                    return None
            
            # Create Dask DataFrame from Parquet
            df = dd.read_parquet(self.parquet_path)
            
            # Apply filters using Dask
            if start_date and end_date:
                df = df[
                    (df['Start_Time'].dt.date >= start_date) &
                    (df['Start_Time'].dt.date <= end_date)
                ]
            
            if severity_range:
                df = df[
                    (df['Severity'] >= severity_range[0]) &
                    (df['Severity'] <= severity_range[1])
                ]
            
            if time_filter and time_filter != "All":
                df = df[df['Sunrise_Sunset'] == time_filter]
            
            # Compute the filtered DataFrame
            filtered_df = df.compute()
            
            # Optimize memory usage
            filtered_df = self.optimize_memory(filtered_df)
            
            # Add color mapping
            filtered_df['color'] = filtered_df['Severity'].map({
                1: '#ffeda0',
                2: '#feb24c',
                3: '#f03b20',
                4: '#bd0026'
            })
            
            # Clean up memory
            gc.collect()
            
            return filtered_df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

def load_accident_data():
    """Optimized function to load and preprocess accident data"""
    try:
        file_path = 'D:\\SRM\\DOMCE\\Project\\accident_data.csv'
        loader = OptimizedDataLoader(file_path)
        
        # Load initial data without filters
        df = loader.load_filtered_data()
        
        if df is None:
            raise Exception("Failed to load data")
        
        return df, loader  # Return both DataFrame and loader instance
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Update main function to use optimized loader
def main():
    st.title("Analysis and Prediction of Road Accidents")
    
    # Load initial data
    df, loader = load_accident_data()
    if df is None or loader is None:
        st.error("Unable to load accident data. Please check your CSV file.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date range filter
    min_date = df['Start_Time'].min().date()
    max_date = df['Start_Time'].max().date()
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Severity filter
    severity_options = sorted(df['Severity'].unique())
    severity_range = st.sidebar.slider(
        "Severity Range", 
        min_value=min(severity_options),
        max_value=max(severity_options),
        value=(min(severity_options), max(severity_options))
    )
    
    # Time of day filter
    time_options = ["All"] + sorted(df['Sunrise_Sunset'].unique().tolist())
    time_filter = st.sidebar.selectbox("Time of Day", time_options)
    
    # Load filtered data using optimized loader
    filtered_df = loader.load_filtered_data(
        start_date=start_date,
        end_date=end_date,
        severity_range=severity_range,
        time_filter=time_filter
    )
    
    if filtered_df is None or filtered_df.empty:
        st.warning("No accidents found with current filters.")
        return
    
    # Show number of filtered accidents
    st.sidebar.metric("Filtered Accidents", len(filtered_df))
    
    # Continue with the rest of the visualization code...
    # [Previous visualization code remains the same]