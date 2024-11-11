import os
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from folium import Map, Marker, PolyLine, Icon
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from opencage.geocoder import OpenCageGeocode
import osmnx as ox
import networkx as nx

# Load environment variables
load_dotenv()
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")

def load_accident_data():
    """Load and preprocess accident data from CSV"""
    try:
        # Read CSV with ISO8601 date parsing, load only first 1000 records
        df = pd.read_csv(
            'D:\\SRM\\DOMCE\\Project\\accident_data.csv', 
            parse_dates=['Start_Time', 'End_Time'], 
            date_format='ISO8601'
        )
        
        # Ensure 'Start_Time' and 'End_Time' are datetime objects
        df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
        df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
        
        # Create severity color mapping (1-4 scale to hex colors)
        df['color'] = df['Severity'].map({
            1: '#ffeda0',  # Light yellow
            2: '#feb24c',  # Orange
            3: '#f03b20',  # Red
            4: '#bd0026'   # Dark red
        })
        
        # Handle any missing coordinates
        df = df.dropna(subset=['Start_Lat', 'Start_Lng'])
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_heatmap(df):
    """Create a heatmap using folium"""
    # Calculate map center based on data
    center_lat = df['Start_Lat'].mean()
    center_lng = df['Start_Lng'].mean()
    
    # Create a folium map
    folium_map = Map(location=[center_lat, center_lng], zoom_start=10)
    
    # Add heatmap layer
    heat_data = [[row['Start_Lat'], row['Start_Lng']] for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(folium_map)
    
    return folium_map

def add_directions(folium_map, start_location, end_location):
    """Add directions from start_location to end_location on the folium map using osmnx"""
    geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
    
    try:
        start_results = geocoder.geocode(start_location)
        end_results = geocoder.geocode(end_location)
        
        if start_results and end_results:
            start_coords = start_results[0]['geometry']
            end_coords = end_results[0]['geometry']
            start_point = (start_coords['lat'], start_coords['lng'])
            end_point = (end_coords['lat'], end_coords['lng'])
            
            # Add markers for start and end points
            Marker(location=start_point, popup="Start", icon=Icon(color='green')).add_to(folium_map)
            Marker(location=end_point, popup="End", icon=Icon(color='red')).add_to(folium_map)
            
            # Get the graph for the area
            G = ox.graph_from_point(start_point, dist=10000, network_type='drive')
            
            # Find the nearest nodes to the start and end points
            start_node = ox.distance.nearest_nodes(G, start_coords['lng'], start_coords['lat'])
            end_node = ox.distance.nearest_nodes(G, end_coords['lng'], end_coords['lat'])
            
            # Find the shortest path
            route = nx.shortest_path(G, start_node, end_node, weight='length')
            
            # Get the coordinates of the route
            route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
            
            # Add the route to the map
            PolyLine(locations=route_coords, color='blue').add_to(folium_map)
            
            # Save route instructions to a text file
            file_path = os.path.join(os.getcwd(), "route_instructions.txt")
            with open(file_path, "w") as file:
                for i, coord in enumerate(route_coords):
                    file.write(f"Step {i+1}: {coord}\n")
            st.success(f"Route instructions saved to {file_path}")
        else:
            st.error("Could not geocode one or both locations. Please check the input.")
    
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
        


def main():
    st.title("Analysis and Prediction of Road Accidents")
    
    # Load data
    df = load_accident_data()
    if df is None or df.empty:
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
    df['Sunrise_Sunset'] = df['Sunrise_Sunset'].astype(str)
    time_options = ["All"] + sorted(df['Sunrise_Sunset'].unique().tolist())
    time_filter = st.sidebar.selectbox("Time of Day", time_options)
    
    # Filter data based on selections
    mask = (
        (df['Start_Time'].dt.date >= start_date) &
        (df['Start_Time'].dt.date <= end_date) &
        (df['Severity'].between(severity_range[0], severity_range[1]))
    )
    
    if time_filter != "All":
        mask &= (df['Sunrise_Sunset'] == time_filter)
    
    filtered_df = df[mask]
    
    # Show number of filtered accidents
    st.sidebar.metric("Filtered Accidents", len(filtered_df))
    
    if filtered_df.empty:
        st.warning("No accidents found with current filters.")
        return
    
    # Create heatmap
    folium_map = create_heatmap(filtered_df)
    
    # Directions input
    st.sidebar.header("Directions")
    start_location = st.sidebar.text_input("Start Location", placeholder="e.g., New York, NY, USA")
    end_location = st.sidebar.text_input("End Location", placeholder="e.g., Boston, MA, USA")
    
    if st.sidebar.button("Get Directions"):
        if start_location and end_location:
            add_directions(folium_map, start_location, end_location)
        else:
            st.sidebar.error("Please enter both start and end locations.")
    
    # Display map
    st.subheader("Accident Heatmap")
    st_data = st_folium(folium_map, width=800, height=600)
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Accidents", len(filtered_df))
    
    with col2:
        avg_severity = filtered_df['Severity'].mean()
        st.metric("Average Severity", f"{avg_severity:.2f}")
    
    with col3:
        most_common_condition = filtered_df['Weather_Condition'].mode().iloc[0]
        st.metric("Most Common Weather", most_common_condition)
    
    # Display data table
    st.subheader("Accident Details")
    display_cols = ['Start_Time', 'Severity', 'Weather_Condition', 'Street', 'City']
    st.dataframe(
        filtered_df[display_cols].sort_values('Start_Time', ascending=False),
        hide_index=True
    )

if __name__ == "__main__":
    main()